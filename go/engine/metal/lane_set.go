// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
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
}

// decodeLane is one lane's owned mutable state.
type decodeLane struct {
	id   int
	sess *ArchSession // owns KV/pos/scratch/ICB; shares weights via the model's shards + residentBufs

	pos          int    // tokens resident in this lane's cache (next token decodes here)
	hidden       []byte // the current post-stack hidden — head+greedy over this yields the next token
	pendingToken int32  // Phase 1's produced token, fed into the Phase 2 batched forward
	hasPLE       bool
	maxNew       int
	generated    int
	stops        map[int32]bool
	terminal     bool
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
// prefills the prompt through the SAME recorded-ICB step the decode uses (so the
// lane's decode caches are populated exactly as the fused Step will read them),
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
	if !samplerIsGreedy(spec.Sampler) {
		return inference.LaneHandle{}, core.NewError("native.laneSet.Prepare: slice-1 owner is greedy only (temperature 0); non-greedy sampling not yet supported")
	}
	if err := ctx.Err(); err != nil {
		return inference.LaneHandle{}, err
	}

	sess, err := ls.openLaneSession()
	if err != nil {
		return inference.LaneHandle{}, err
	}
	icb := sess.state.icb
	if icb == nil || !icb.hasFinalOut {
		_ = sess.Close()
		return inference.LaneHandle{}, core.NewError("native.laneSet.Prepare: model is not recorded-ICB eligible — multi-session batched step needs the dense/PLE recorded-ICB decode path")
	}
	if ls.dModel == 0 {
		ls.dModel = sess.arch.Hidden
	}
	if len(spec.PromptIDs) > sess.maxLen {
		_ = sess.Close()
		return inference.LaneHandle{}, core.NewError("native.laneSet.Prepare: prompt exceeds model context window")
	}

	lane := &decodeLane{
		id:     ls.nextID + 1,
		sess:   sess,
		hasPLE: sess.perLayerInput != nil,
		maxNew: spec.MaxNew,
		stops:  buildStopSet(spec.StopTokens),
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

// prefillLane feeds the prompt through the lane's recorded ICB one token at a
// time — the same stepBody the fused decode replays, so the lane's caches are
// populated identically to how the decode reads them (no dependence on which of
// the several production prefill routes a batched prefill would have taken).
// Batched prefill admission is the pinned slice-2 next rung.
func (ls *laneSet) prefillLane(lane *decodeLane, promptIDs []int32) error {
	icb := lane.sess.state.icb
	var perr error
	withAutoreleasePool(func() {
		for i, id := range promptIDs {
			emb, err := lane.sess.embedID(id)
			if err != nil {
				perr = err
				return
			}
			var pli []byte
			if lane.hasPLE {
				if pli, err = lane.sess.perLayerInput(id, emb); err != nil {
					perr = err
					return
				}
				lane.sess.state.perLayerInput = pli
			}
			if i == len(promptIDs)-1 {
				lane.hidden = append(lane.hidden[:0], icb.stepBody(emb, lane.pos, pli)...)
			} else {
				icb.stepBodyNoResult(emb, lane.pos, pli)
			}
			lane.pos++
		}
	})
	return perr
}

// Step advances every active, non-terminal lane by one token through ONE shared
// command buffer. Phase 1 produces this round's token per lane (head + greedy on
// the lane's current hidden — the same per-lane op the serial loop runs). Phase 2
// feeds each still-live lane's just-produced token into its recorded ICB, all
// replayed into a single encoder, committed and waited ONCE — the batched
// forward that advances the whole set to its next hidden.
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
		// Phase 1 — produce one token per active lane from its current hidden.
		for _, lane := range active {
			tok, err := lane.sess.greedyFromHiddenInPool(lane.hidden, nil)
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
		// Phase 2 — ONE batched forward that advances the whole set. The
		// weight-read-once GEMM forward (lane_set_gemm.go) sweeps each weight once
		// for all K lanes; LTHN_CB_GEMM=0 or an ineligible arch falls back to the
		// per-lane ICB replay (byte-for-byte the merged 2.58× path).
		if ls.gemmForwardEnabled() && ls.gemmEligible(advancing) {
			if err := ls.batchedGEMMForward(advancing); err != nil {
				stepErr = err
			}
			return
		}
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		for _, lane := range advancing {
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
		for _, lane := range advancing {
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

// samplerIsGreedy reports whether cfg selects greedy (temperature-0) decoding —
// the only discipline slice 1 serves. The zero value is greedy.
func samplerIsGreedy(cfg inference.SamplerConfig) bool {
	return cfg.Temperature == 0 && cfg.TopK == 0 && cfg.TopP == 0 && cfg.MinP == 0 && cfg.RepeatPenalty == 0
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
