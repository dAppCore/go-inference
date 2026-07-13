// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model"
	"github.com/tmc/apple/metal"
)

// laneSet is the metal multi-session state owner: K decode lanes that SHARE the
// model's immutable weights (each lane is an ArchSession opened from the same
// NativeTokenModel, so the resident weight buffers — keyed by mmap address in
// the process-global residentBufs cache — are the SAME device buffers for every
// lane), while each lane OWNS its mutable decode state (its recorded-ICB decode
// caches, its scalar position, its per-lane embed/PLE/greedy scratch). One Step
// advances every active lane by exactly one token through ONE shared command
// buffer: each lane's recorded ICB is replayed into a single encoder and the
// whole batch is committed and waited ONCE, so K lanes cost one GPU submission
// (host dispatch amortised K-fold, and the GPU may pipeline the lanes' disjoint
// executions) instead of K separate commit/wait round-trips.
//
// Byte-identity: a lane replays its OWN recorded ICB over its OWN caches at its
// OWN position — the exact GPU commands ArchSession.Step runs for that lane in
// isolation. The lanes touch disjoint device buffers (only the read-only
// weights are shared), so fusing them into one command buffer changes
// submission and scheduling, never arithmetic. The counter-guarded conformance
// fixture proves this: the same lane specs produce the same per-lane token
// streams whether each is run alone (K==1) or all together (K>1), and
// BatchForwardCount advances by the number of Steps (one batched forward per
// Step), not by Steps×K (which K separate single-session steps would give).
//
// Scope (slice 1): the recorded-ICB dense/PLE decode path (gemma4 dense, E2B/
// E4B) and greedy (temperature-0) sampling. A non-ICB arch (MoE 12B/26B,
// COMPOSED/hybrid) or a non-greedy sampler is refused rather than served on a
// path that could not be proven byte-identical here — see docs/design-
// continuous-batching.md for the pinned next rungs (batched-projection weight-
// read-once GEMMs; non-ICB arches; batched head/prefill).
//
// A laneSet is single-goroutine, exactly as an ArchSession is: the step
// coordinator that owns it drives Prepare/Step/Retire from one goroutine.
type laneSet struct {
	model    *NativeTokenModel
	maxLanes int
	dModel   int
	vocab    int

	lanes  map[int]*decodeLane
	order  []int // stable admission order → deterministic Step iteration
	nextID int

	fwdCount uint64 // monotonic: +1 per batched forward (one per Step that advances ≥1 lane)

	// gemmMode caches the LTHN_CB_GEMM read (0 unread, 1 armed, 2 disabled); gemm
	// holds the reusable K-row projection staging (lane_set_gemm.go).
	gemmMode     int
	gemm         *gemmSlabs
	gemmFwdCount uint64 // monotonic: +1 per weight-read-once GEMM forward (the counter guard)

	// headRowsCount is monotonic: +1 per batched Phase-1 head submission (the
	// K-row fused lm_head+argmax). The engagement discriminator for tests — a
	// silent fall-through to the per-lane ladder must not pass as batched.
	headRowsCount uint64
}

// decodeLane is one lane's owned mutable state.
type decodeLane struct {
	id   int
	sess *ArchSession // owns KV/pos/scratch/ICB; shares weights via the model's shards + residentBufs

	pos          int    // tokens resident in this lane's cache (next token decodes here)
	hidden       []byte // the current post-stack hidden — the per-lane head over this yields the next token
	pendingToken int32  // Phase 1's produced token, fed into the Phase 2 batched forward
	hasPLE       bool
	maxNew       int
	generated    int
	stops        map[int32]bool
	terminal     bool
	// reencode marks a lane whose session has no recorded ICB (MoE — the host
	// router / device-router block is not recorder-supported): Phase 2 advances
	// it through the session's own one-token re-encode step (stepIDInPool)
	// instead of the shared ICB replay. Byte-identical to the plain path by
	// construction — it IS the plain path's step.
	reencode bool

	// Non-greedy discipline (lane_set_sampling.go): the lane's OWN sampler RNG
	// stream + params + repeat-penalty history; nil sampler = greedy phase 1.
	sampler       *model.Sampler
	sampleParams  model.SampleParams
	sampleHistory []int32
}

var _ inference.LaneSet = (*laneSet)(nil)

// OpenLaneSet builds the multi-session owner over this model's shared weights.
// It is the engine.LaneSetOpener capability engine.TextModel surfaces to the
// neutral inference.BatchStepModel contract. The kill switch (LTHN_CB_STEP=0)
// is enforced one layer up, in engine.TextModel, so it is not re-checked here.
func (m *NativeTokenModel) OpenLaneSet(cfg inference.LaneSetConfig) (inference.LaneSet, error) {
	if m == nil {
		return nil, core.NewError("native.OpenLaneSet: nil model")
	}
	maxLanes := cfg.MaxLanes
	if maxLanes <= 0 {
		maxLanes = defaultLaneSetMaxLanes
	}
	return &laneSet{
		model:    m,
		maxLanes: maxLanes,
		vocab:    m.vocab,
		lanes:    make(map[int]*decodeLane, maxLanes),
	}, nil
}

// defaultLaneSetMaxLanes bounds concurrent lanes when the caller gives no cap.
// Conservative: KV co-residency for several long-context sessions is a real
// device-memory user (docs/design-continuous-batching.md §c), and slice 1 does
// not yet gate admission through kv/budget.FitsMemory.
const defaultLaneSetMaxLanes = 8

// Prepare admits a new lane: it opens a fresh session sharing the model weights,
// prefills the prompt through the session's production prefill route (so the
// lane's decode caches are populated exactly as the plain path populates them),
// and leaves the lane holding its prefill hidden — ready to produce its first
// token on the next Step. Ragged admission: safe to call between Steps.
func (ls *laneSet) Prepare(ctx context.Context, spec inference.LaneSpec) (inference.LaneHandle, error) {
	if ls == nil || ls.model == nil {
		return inference.LaneHandle{}, core.NewError("native.laneSet.Prepare: nil lane set")
	}
	if len(ls.lanes) >= ls.maxLanes {
		return inference.LaneHandle{}, core.NewError("native.laneSet.Prepare: lane set is at MaxLanes")
	}
	if len(spec.PromptIDs) == 0 {
		return inference.LaneHandle{}, core.NewError("native.laneSet.Prepare: empty prompt")
	}
	if spec.MaxNew <= 0 {
		return inference.LaneHandle{}, core.NewError("native.laneSet.Prepare: MaxNew must be > 0")
	}
	if err := ctx.Err(); err != nil {
		return inference.LaneHandle{}, err
	}

	sess, err := ls.openLaneSession()
	if err != nil {
		return inference.LaneHandle{}, err
	}
	// A session without a recorded ICB (MoE — icbEligible declines the router
	// block) is admitted as a RE-ENCODE lane: Phase 2 advances it through the
	// session's own one-token step rather than the shared ICB replay. The set
	// still owns admission, the batched Phase-1 head, ragged retirement and
	// per-request accounting — the shared-submission win stays ICB-only.
	icb := sess.state.icb
	reencode := icb == nil || !icb.hasFinalOut
	if ls.dModel == 0 {
		ls.dModel = sess.arch.Hidden
	}
	if len(spec.PromptIDs) > sess.maxLen {
		_ = sess.Close()
		return inference.LaneHandle{}, core.NewError("native.laneSet.Prepare: prompt exceeds model context window")
	}

	lane := &decodeLane{
		id:       ls.nextID + 1,
		sess:     sess,
		hasPLE:   sess.perLayerInput != nil,
		maxNew:   spec.MaxNew,
		stops:    buildStopSet(spec.StopTokens),
		reencode: reencode,
	}
	if laneSampled(spec.Sampler) {
		lane.sampler = model.NewSampler(spec.SampleSeed)
		lane.sampleParams = laneSampleParams(spec.Sampler)
	}
	if err := ls.prefillLane(lane, spec.PromptIDs); err != nil {
		_ = sess.Close()
		return inference.LaneHandle{}, err
	}

	ls.nextID++
	ls.lanes[lane.id] = lane
	ls.order = append(ls.order, lane.id)
	return inference.LaneHandle{ID: lane.id}, nil
}

// prefillLane runs the lane's prompt through the session's PRODUCTION prefill
// route (prefillRetainedTokens: chunked batched-dense forward, flash prompt
// SDPA, the kv-share skip — everything the plain serve path prefills with)
// and takes the boundary hidden as the lane's decode seed. This replaced the
// original one-token-at-a-time stepBody loop: byte-compatible (the sampled
// and E2B oracle receipts compare lanes against classic sessions prefilled by
// exactly this route) but ~40× cheaper on admission — the serial loop paid
// one commit+wait per prompt token (measured live 2026-07-13: 4 concurrent
// 1161-token chat admissions took 29.8s serial vs 0.6s on this route).
func (ls *laneSet) prefillLane(lane *decodeLane, promptIDs []int32) error {
	hidden, err := lane.sess.prefillRetainedTokens(promptIDs, "native.laneSet.Prepare")
	if err != nil {
		return err
	}
	if len(hidden) == 0 {
		return core.NewError("native.laneSet.Prepare: prefill returned no boundary hidden")
	}
	lane.hidden = append(lane.hidden[:0], hidden...)
	lane.pos = lane.sess.pos
	return nil
}

// Step advances every active, non-terminal lane by one token. Phase 1 produces
// this round's token per lane (head + greedy/sampled on the lane's current
// hidden — the same per-lane op the serial loop runs). Phase 2 feeds each
// still-live recorded-ICB lane's just-produced token into its ICB, all replayed
// into a single encoder, committed and waited ONCE — the batched forward that
// advances the whole set to its next hidden; re-encode lanes (MoE) advance
// through their session's own one-token step instead.
func (ls *laneSet) Step(ctx context.Context) ([]inference.LaneStep, error) {
	if ls == nil {
		return nil, core.NewError("native.laneSet.Step: nil lane set")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	active := ls.activeLanes()
	if len(active) == 0 {
		return nil, nil
	}

	results := make([]inference.LaneStep, 0, len(active))
	var advancing []*decodeLane
	var stepErr error
	withAutoreleasePool(func() {
		// Phase 1 — produce one token per active lane from its current hidden:
		// the lane's own sampler for a non-greedy discipline (the classic sampled
		// route ladder, one draw per token), else head+greedy. An ALL-GREEDY set
		// takes ONE batched head submission (the MTP verify's K-row fused
		// lm_head+argmax — one weight sweep for all K on quant heads) instead of
		// K commit+waits; any decline falls to the per-lane ladder.
		batchedToks, batchedHead := ls.phase1GreedyRows(active)
		for i, lane := range active {
			var tok int32
			var err error
			switch {
			case batchedHead:
				tok = batchedToks[i]
			case lane.sampler != nil:
				tok, err = lane.sess.sampledNextFromHiddenInPool(lane.hidden, lane.sampler, lane.sampleParams, lane.sampleHistory)
				if err == nil && lane.sampleParams.RepeatPenalty > 1 {
					lane.sampleHistory = append(lane.sampleHistory, tok)
				}
			default:
				tok, err = lane.sess.greedyFromHiddenInPool(lane.hidden, nil)
			}
			if err != nil {
				stepErr = err
				return
			}
			lane.generated++
			terminal := lane.stops[tok] || lane.generated >= lane.maxNew
			if terminal {
				lane.terminal = true
			}
			results = append(results, inference.LaneStep{
				Lane:     inference.LaneHandle{ID: lane.id},
				Token:    tok,
				HasToken: true,
				Terminal: terminal,
			})
			if !terminal {
				lane.pendingToken = tok
				advancing = append(advancing, lane)
			}
		}
		if len(advancing) == 0 {
			return
		}
		// Phase 2 partition: recorded-ICB lanes ride the shared replay (fold or
		// one-encoder batch); re-encode lanes (MoE) advance through the plain
		// path's per-token machinery — K forwards encoded into ONE shared
		// submission when every session is shared-encode eligible (the 26B
		// device-router lane), else each lane's own step. One model per set
		// makes the partition all-or-nothing in practice; the split keeps a
		// hybrid correct anyway.
		var replay, reenc []*decodeLane
		for _, lane := range advancing {
			if lane.reencode {
				reenc = append(reenc, lane)
			} else {
				replay = append(replay, lane)
			}
		}
		if len(reenc) > 0 && !ls.sharedReencodeForward(reenc) {
			for _, lane := range reenc {
				h, err := lane.sess.stepIDInPool(lane.pendingToken)
				if err != nil {
					stepErr = err
					return
				}
				lane.hidden = append(lane.hidden[:0], h...)
				lane.pos = lane.sess.pos
			}
		}
		if len(replay) == 0 {
			return
		}
		// ONE batched forward that advances the whole replay set. The
		// weight-read-once GEMM forward (lane_set_gemm.go) sweeps each weight once
		// for all K lanes; LTHN_CB_GEMM=0 or an ineligible arch falls back to the
		// per-lane ICB replay (byte-for-byte the merged 2.58× path).
		if ls.gemmForwardEnabled() && ls.gemmEligible(replay) &&
			(ls.gemmMode == 1 || ls.gemmProfitable(replay)) {
			if err := ls.batchedGEMMForward(replay); err != nil {
				stepErr = err
			}
			return
		}
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		for _, lane := range replay {
			emb, err := lane.sess.embedID(lane.pendingToken)
			if err != nil {
				endEncodingFast(enc)
				stepErr = err
				return
			}
			var pli []byte
			if lane.hasPLE {
				if pli, err = lane.sess.perLayerInput(lane.pendingToken, emb); err != nil {
					endEncodingFast(enc)
					stepErr = err
					return
				}
				lane.sess.state.perLayerInput = pli
			}
			lane.sess.state.icb.encodeStepBody(enc, emb, lane.pos, pli)
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		ls.fwdCount++
		// Read each advanced lane's new hidden and move its position on.
		for _, lane := range replay {
			icb := lane.sess.state.icb
			if cap(lane.hidden) < ls.dModel*bf16Size {
				lane.hidden = make([]byte, ls.dModel*bf16Size)
			}
			lane.hidden = lane.hidden[:ls.dModel*bf16Size]
			icb.copyLastOutInto(lane.hidden)
			lane.pos++
		}
	})
	if stepErr != nil {
		return nil, stepErr
	}
	return results, nil
}

// phase1GreedyRows runs Phase 1's head for an ALL-GREEDY active set as one
// batched submission: every lane's hidden rides the K-row fused
// lm_head+argmax (ArchSession.greedyRowsFromHiddensInPool — the MTP verify
// head), one command buffer, one wait, and on quant heads ONE weight sweep
// scoring all K rows. The rows buffer and head encoder are shared model
// state, so ANY lane's session can host the call. ok=false — a sampled lane
// in the set, K==1 (nothing to batch), no direct-greedy head, or the
// helper's own declines — keeps the per-lane ladder, byte-identically.
func (ls *laneSet) phase1GreedyRows(active []*decodeLane) ([]int32, bool) {
	if len(active) < 2 {
		return nil, false
	}
	for _, lane := range active {
		if lane.sampler != nil {
			return nil, false
		}
	}
	hiddens := make([][]byte, len(active))
	for i, lane := range active {
		hiddens[i] = lane.hidden
	}
	out := make([]int32, len(active))
	ok, err := active[0].sess.greedyRowsFromHiddensInPool(hiddens, nil, out)
	if err != nil || !ok {
		return nil, false
	}
	ls.headRowsCount++
	return out, true
}

// sharedReencodeForward advances the re-encode lanes through ONE grouped
// submission round: each lane's full one-token forward encodes into its OWN
// command buffer, all K are committed back-to-back, and only then does the
// owner wait — the waits pipeline (the first absorbs the bulk; the rest are
// already near-complete), and the queue overlaps the K independent cbs
// exactly as the plain path's per-request goroutines do. One cb for ALL K
// was measured slower here: the step's pass-edge memory barriers are global
// to the encoder chain, so a single cb serialises the lanes against each
// other and forfeits the overlap it was meant to buy. Returns false WITHOUT
// advancing any lane when a session is ineligible (bf16/host-router MoE,
// trace, PLE) or an encode declines mid-way — the caller then runs the
// per-lane path; uncommitted cbs mutate no session state, so the retry is
// clean. On success every lane's hidden and position are advanced and the
// step counts one batched forward round.
func (ls *laneSet) sharedReencodeForward(reenc []*decodeLane) bool {
	for _, lane := range reenc {
		if !lane.sess.sharedStepEligible() {
			return false
		}
	}
	cbs := make([]metal.MTLCommandBufferObject, len(reenc))
	finalOut := make([]metal.MTLBuffer, len(reenc))
	for i, lane := range reenc {
		sink := &sharedStepSink{cb: commandBufferFast(queue)}
		sink.enc = computeCommandEncoderFast(sink.cb)
		if err := lane.sess.stepIDEncodeShared(lane.pendingToken, sink); err != nil {
			// The failing step already ended its encoder; nothing committed
			// yet on THIS lane — but earlier lanes' cbs are committed below
			// only after every encode succeeds, so drop everything and let
			// the caller retry per-lane.
			return false
		}
		endEncodingFast(sink.enc)
		cbs[i], finalOut[i] = sink.cb, sink.finalOut
	}
	for _, cb := range cbs {
		commitCommandBufferFast(cb)
	}
	for _, cb := range cbs {
		waitUntilCompletedFast(cb)
	}
	ls.fwdCount++
	for i, lane := range reenc {
		if cap(lane.hidden) < ls.dModel*bf16Size {
			lane.hidden = make([]byte, ls.dModel*bf16Size)
		}
		lane.hidden = lane.hidden[:ls.dModel*bf16Size]
		copy(lane.hidden, lane.sess.state.bufferBytes(finalOut[i], ls.dModel*bf16Size))
		lane.sess.pos++
		lane.pos = lane.sess.pos
	}
	return true
}

// activeLanes returns the non-terminal lanes in stable admission order.
func (ls *laneSet) activeLanes() []*decodeLane {
	out := make([]*decodeLane, 0, len(ls.order))
	for _, id := range ls.order {
		if lane := ls.lanes[id]; lane != nil && !lane.terminal {
			out = append(out, lane)
		}
	}
	return out
}

// Retire removes a lane and releases its per-lane decode state. The shared
// weights stay resident (they belong to the model).
func (ls *laneSet) Retire(h inference.LaneHandle) error {
	if ls == nil {
		return core.NewError("native.laneSet.Retire: nil lane set")
	}
	lane, ok := ls.lanes[h.ID]
	if !ok {
		return core.NewError("native.laneSet.Retire: unknown lane")
	}
	delete(ls.lanes, h.ID)
	for i, id := range ls.order {
		if id == h.ID {
			ls.order = append(ls.order[:i], ls.order[i+1:]...)
			break
		}
	}
	return lane.sess.Close()
}

// Active reports the number of admitted, non-retired lanes.
func (ls *laneSet) Active() int {
	if ls == nil {
		return 0
	}
	return len(ls.lanes)
}

// BatchForwardCount is the monotonic count of batched forwards Step has run.
func (ls *laneSet) BatchForwardCount() uint64 {
	if ls == nil {
		return 0
	}
	return ls.fwdCount
}

// Close retires every remaining lane and releases the owner.
func (ls *laneSet) Close() error {
	if ls == nil {
		return nil
	}
	var firstErr error
	for _, lane := range ls.lanes {
		if err := lane.sess.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	ls.gemm.release()
	ls.gemm = nil
	ls.lanes = nil
	ls.order = nil
	return firstErr
}

// openLaneSession opens a fresh ArchSession sharing the model's weights (shards
// + resident buffers) — the same construction OpenSession/OpenEngineSession use.
func (ls *laneSet) openLaneSession() (*ArchSession, error) {
	stepper, err := ls.model.openSession(ls.model.shards, ls.model.headEnc)
	if err != nil {
		return nil, err
	}
	sess, ok := stepper.(*ArchSession)
	if !ok {
		if closer, ok := stepper.(interface{ Close() error }); ok {
			_ = closer.Close()
		}
		return nil, core.NewError("native.laneSet: engine session is not a recorded-ICB ArchSession")
	}
	return sess, nil
}

// buildStopSet indexes the request's stop tokens. The caller resolves any
// model-declared stops (EOS, turn-close, template stops) into spec.StopTokens
// exactly as the serve layer already builds its stop set, so the engine owner
// stays free of template knowledge.
func buildStopSet(reqStops []int32) map[int32]bool {
	if len(reqStops) == 0 {
		return nil
	}
	stops := make(map[int32]bool, len(reqStops))
	for _, t := range reqStops {
		stops[t] = true
	}
	return stops
}
