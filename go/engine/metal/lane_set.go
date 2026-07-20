// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"
	"os"
	"sync/atomic"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/kv/budget"
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
// Scope: the recorded-ICB dense/PLE decode path (gemma4 dense, E2B/E4B) rides
// the shared ICB replay (fold or one-encoder batch — the shared-submission
// win). A non-ICB arch (MoE 12B/26B; the router block is not ICB-recordable) is
// admitted as a RE-ENCODE or CHAINED lane and advanced through the session's
// own one-token step — byte-identical (it IS the plain path's step), just
// without the ICB shared-submission fold. Greedy AND per-lane non-greedy
// sampling are both served (lane_set_sampling.go: each lane owns its sampler
// RNG + params). Prepare refuses only an empty, over-length or cancelled
// prompt — never an architecture.
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

	// liveLanes mirrors len(lanes) atomically — the ONE laneSet datum an
	// off-goroutine BeginPrepare may read (advisory: it picks the overlapped
	// prefill's chunk cap, never a correctness decision). Written only by the
	// owning goroutine (CommitPrepare / Retire / Close).
	liveLanes atomic.Int64

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
	// chainedSteps is monotonic: +1 per chained forward round (the fused
	// forward+head+embed lane submissions) — the chained path's engagement
	// discriminator.
	chainedSteps uint64
	// sampledRowsCount is monotonic: +1 per batched sampled Phase-1
	// submission (K topK-token chains in one command buffer) — the sampled
	// batch's engagement discriminator.
	sampledRowsCount uint64
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

	// Chained lane state (greedy re-encode lanes only): each round's forward
	// carries the head argmax + the next token's embed gather in the SAME
	// command buffer — the serial chained-live decode's shape, per lane —
	// and free-runs RAGGED: one COMMITTED round in flight (inFlight*) plus
	// one PRE-ENCODED round held UNCOMMITTED behind it (held*). A Step polls
	// completion per lane and, the instant a lane's round finishes, reads its
	// 4-byte token, then COMMITS the held round (µs — the lane rolls on
	// without waiting any other lane or the Step boundary) and pre-encodes
	// the next held one overlapped with the new in-flight round. A stop
	// token just DROPS the held round un-committed: zero speculation waste,
	// no position rewind, no KV truncation.
	inFlightCB   metal.MTLCommandBufferObject
	inFlightScr  *headGreedyScratch
	inFlightLive bool
	heldCB       metal.MTLCommandBufferObject
	heldScr      *headGreedyScratch
	heldLive     bool
	needAdvance  bool          // exited the chain with an emitted-but-unadvanced token (defensive)
	chainSc      *plGPUScratch // next-inputs gather scratch (zero-value for non-PLE)
	chainDead    bool          // the head declined once — the lane stays two-phase

	// Non-greedy discipline (lane_set_sampling.go): the lane's OWN sampler RNG
	// stream + params + repeat-penalty history; nil sampler = greedy phase 1.
	sampler       *model.Sampler
	sampleParams  model.SampleParams
	sampleHistory []int32
}

var _ inference.LaneSet = (*laneSet)(nil)
var _ inference.LaneSetOverlappedAdmitter = (*laneSet)(nil)

// OpenLaneSet builds the multi-session owner over this model's shared weights.
// It is the engine.LaneSetOpener capability engine.TextModel surfaces to the
// neutral inference.BatchStepModel contract. The kill switch (LTHN_CB_STEP=0)
// is enforced one layer up, in engine.TextModel, so it is not re-checked here.
func (m *NativeTokenModel) OpenLaneSet(cfg inference.LaneSetConfig) (inference.LaneSet, error) {
	if m == nil {
		return nil, core.NewError("native.OpenLaneSet: nil model")
	}
	// TurboQuant live KV declines the batch/interleave lanes (v1): the laneSet's
	// interleaved landings and GEMM folds read/write bf16 cache rows. Refuse
	// loudly — the per-session decode lane carries TQ.
	if tqMode, _ := parseTurboQuantCacheMode(m.kvCacheMode); tqMode != nil {
		return nil, core.NewError("native.OpenLaneSet: -kv-cache turboquant declines the continuous-batching laneSet (v1) — serve per-session or drop -kv-cache")
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
// device-memory user (docs/design-continuous-batching.md §c). This count alone
// does not know a lane's KV cost, so admission additionally gates through
// kv/budget.FitsMemory (admitMemoryBudget) — a full-but-tiny MaxLanes cap must
// not let K long-context lanes OOM a box this count alone would have allowed.
const defaultLaneSetMaxLanes = 8

// admitLaneMemoryBudget is the pure decision behind admitMemoryBudget: does
// admitting the liveLanes-th lane (this candidate included) keep the model's
// total resident KV bytes within the device memory budget? Every lane opened
// from the same laneSet shares one fixed context length and architecture
// (NativeBackend.arch/maxLen, captured once at load and closed over by
// NativeTokenModel.openSession — every ArchSession a laneSet opens reuses
// them), so one lane's footprint (sessionKVBytesAt) times the lane count IS
// the set's exact KV total.
//
// It reuses kv/budget.FitsMemory — the same fits-a-budget primitive the
// single-session load path's RAM guard is built from (clampContextToRAM,
// same file) — handing it the admission's total byte requirement as ONE
// "token" of that many "bytes": FitsMemory's
// workingTokens*bytesPerToken<=deviceBudget then performs the exact
// comparison, rather than a second hand-rolled <=.
//
// ramBytes/weightsBytes are passed in (rather than read here) so this stays a
// pure function of its inputs — admitMemoryBudget is the thin wrapper that
// gathers them from the live model/box, exactly as clampContextToRAM /
// clampDefaultContextToRAM split the same way. Fails OPEN (nil error) when
// ramBytes is unmeasured (0), liveLanes is non-positive (nothing admitted
// yet), or the model prices out to zero KV bytes (no cache-owning layers) —
// never a silent admit beyond that: a measured, over-budget admission returns
// a clean, non-nil error.
func admitLaneMemoryBudget(arch model.Arch, ctxLen int, weightsBytes, ramBytes uint64, liveLanes int64) error {
	if ramBytes == 0 || liveLanes <= 0 {
		return nil
	}
	perLane := sessionKVBytesAt(arch, ctxLen)
	if perLane == 0 {
		return nil
	}
	deviceBudget, ok := sessionMemoryBudgetBytes(weightsBytes, ramBytes)
	working := uint64(liveLanes) * perLane
	if !ok || !budget.FitsMemory(1, intFromBytes(working), intFromBytes(deviceBudget)) {
		return core.NewError(core.Sprintf(
			"native.laneSet: admitting lane %d needs %d MiB of KV cache against a %d MiB device memory budget (weights %d MiB, RAM %d MiB) — declined",
			liveLanes, working>>20, deviceBudget>>20, weightsBytes>>20, ramBytes>>20))
	}
	return nil
}

// admitMemoryBudget is admitLaneMemoryBudget's live wrapper: it gathers this
// laneSet's model geometry, resident weight bytes, and the box's physical RAM,
// then defers the actual fits-or-declines call to the pure function. liveLanes
// is the lane count AFTER this admission (existing lanes + the candidate),
// supplied by the caller because it is read differently depending on which
// goroutine is asking — see the two call sites in BeginPrepare/CommitPrepare.
//
// Fails OPEN (never declines) when the RAM guard is disabled
// (LTHN_CONTEXT_RAM_GUARD=0) — the same kill switch clampContextToRAM honours,
// so both admission paths share one lever.
func (ls *laneSet) admitMemoryBudget(liveLanes int64) error {
	if !contextRAMGuardEnabled || ls == nil || ls.model == nil || ls.model.NativeBackend == nil {
		return nil
	}
	return admitLaneMemoryBudget(ls.model.arch, ls.model.maxLen, ls.model.shards.totalMappedBytes(), physicalRAMBytes(), liveLanes)
}

// Prepare admits a new lane: it opens a fresh session sharing the model weights,
// prefills the prompt through the session's production prefill route (so the
// lane's decode caches are populated exactly as the plain path populates them),
// and leaves the lane holding its prefill hidden — ready to produce its first
// token on the next Step. Ragged admission: safe to call between Steps. It is
// the single-goroutine composition of BeginPrepare + CommitPrepare (the
// overlapped-admission split), with a fail-fast MaxLanes check up front so a
// full set never pays a prefill it must throw away.
func (ls *laneSet) Prepare(ctx context.Context, spec inference.LaneSpec) (inference.LaneHandle, error) {
	if ls == nil || ls.model == nil {
		return inference.LaneHandle{}, core.NewError("native.laneSet.Prepare: nil lane set")
	}
	if len(ls.lanes) >= ls.maxLanes {
		return inference.LaneHandle{}, core.NewError("native.laneSet.Prepare: lane set is at MaxLanes")
	}
	pending, err := ls.BeginPrepare(ctx, spec)
	if err != nil {
		return inference.LaneHandle{}, err
	}
	return ls.CommitPrepare(pending)
}

// pendingLane is the metal inference.PendingLane: a fully prefilled decodeLane
// awaiting its splice into the set (CommitPrepare) or release (Discard).
type pendingLane struct {
	lane   *decodeLane
	dModel int
}

// Discard releases the pending lane's session without attaching it.
func (p *pendingLane) Discard() {
	if p == nil || p.lane == nil {
		return
	}
	_ = p.lane.sess.Close()
	p.lane = nil
}

// BeginPrepare runs the heavy half of admission — session open + the prompt's
// production chunked prefill — WITHOUT touching the set's lane bookkeeping, so
// it is safe to run from another goroutine while the owning goroutine keeps
// Stepping: the pending lane's session is independent state, the same isolation
// that lets concurrent plain-path generations coexist with a stepping lane set
// (the continuity handoff's proven-live pattern). The MaxLanes bound is
// CommitPrepare's to enforce (this side cannot read the lane map racelessly);
// callers budget their in-flight BeginPrepares against the same slot count.
func (ls *laneSet) BeginPrepare(ctx context.Context, spec inference.LaneSpec) (inference.PendingLane, error) {
	if ls == nil || ls.model == nil {
		return nil, core.NewError("native.laneSet.Prepare: nil lane set")
	}
	if len(spec.PromptIDs) == 0 {
		return nil, core.NewError("native.laneSet.Prepare: empty prompt")
	}
	if spec.MaxNew <= 0 {
		return nil, core.NewError("native.laneSet.Prepare: MaxNew must be > 0")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Advisory fail-fast: liveLanes is the ONE laneSet datum safe to read from
	// an off-goroutine BeginPrepare (overlapped admission, struct doc above) —
	// a concurrent BeginPrepare racing this one can still slip past both
	// checks, but CommitPrepare's raceless recheck is the authoritative gate
	// (same shape as the MaxLanes bound this comment already describes), so a
	// slip here only costs a wasted prefill, never a bad admission. Checked
	// before openLaneSession so a doomed admission doesn't even pay for that.
	if err := ls.admitMemoryBudget(ls.liveLanes.Load() + 1); err != nil {
		return nil, err
	}

	sess, err := ls.openLaneSession()
	if err != nil {
		return nil, err
	}
	// Joining a set that is ALREADY streaming: cap the newcomer's prefill
	// chunks so the shared command queue interleaves lane rounds between them
	// — one whole-prompt command buffer would freeze every in-flight stream
	// for its full duration (the queue executes buffers in order). An empty
	// set keeps the uncapped whole-prompt prefill (burst admissions and solo
	// boots lose nothing). Advisory read; byte-identical either way (the #381
	// chunk-parity receipts — chunked ≡ one-shot ≡ stepping).
	if live := ls.liveLanes.Load(); live > 0 {
		sess.prefillChunkRowsCap = laneOverlapPrefillChunkRows
		if gpuTraceEnabled() {
			nativeTraceLog(core.Sprintf("gpu-trace: lane  begin-prepare  live=%d cap=%d rows=%d\n", live, sess.prefillChunkRowsCap, len(spec.PromptIDs)))
		}
	} else if gpuTraceEnabled() {
		nativeTraceLog(core.Sprintf("gpu-trace: lane  begin-prepare  live=0 cap=0 rows=%d\n", len(spec.PromptIDs)))
	}
	// A session without a recorded ICB (MoE — icbEligible declines the router
	// block) is admitted as a RE-ENCODE lane: Phase 2 advances it through the
	// session's own one-token step rather than the shared ICB replay. The set
	// still owns admission, the batched Phase-1 head, ragged retirement and
	// per-request accounting — the shared-submission win stays ICB-only.
	icb := sess.state.icb
	reencode := icb == nil || !icb.hasFinalOut
	if len(spec.PromptIDs) > sess.maxLen {
		_ = sess.Close()
		return nil, core.NewError("native.laneSet.Prepare: prompt exceeds model context window")
	}

	lane := &decodeLane{
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
		return nil, err
	}
	return &pendingLane{lane: lane, dModel: sess.arch.Hidden}, nil
}

// CommitPrepare splices a prepared lane into the running set — id assignment +
// bookkeeping only, no GPU work — and MUST run on the owning goroutine (it
// mutates the lane map Step iterates). At MaxLanes the pending lane is
// released and an error returned; the scheduler's slot budget keeps that from
// happening in practice.
func (ls *laneSet) CommitPrepare(p inference.PendingLane) (inference.LaneHandle, error) {
	if ls == nil {
		return inference.LaneHandle{}, core.NewError("native.laneSet.CommitPrepare: nil lane set")
	}
	pl, ok := p.(*pendingLane)
	if !ok || pl == nil || pl.lane == nil {
		return inference.LaneHandle{}, core.NewError("native.laneSet.CommitPrepare: not a pending metal lane")
	}
	if len(ls.lanes) >= ls.maxLanes {
		pl.Discard()
		return inference.LaneHandle{}, core.NewError("native.laneSet.Prepare: lane set is at MaxLanes")
	}
	// Authoritative recheck: CommitPrepare owns the goroutine and ls.lanes is
	// raceless here, so this is the real gate — BeginPrepare's own check above
	// is only an advisory fail-fast that a concurrent overlapped admission can
	// race past. A candidate that no longer fits is declined and discarded
	// (its session closes; nothing was spliced in, nothing to unwind).
	if err := ls.admitMemoryBudget(int64(len(ls.lanes)) + 1); err != nil {
		pl.Discard()
		return inference.LaneHandle{}, err
	}
	if ls.dModel == 0 {
		ls.dModel = pl.dModel
	}
	lane := pl.lane
	pl.lane = nil
	ls.nextID++
	lane.id = ls.nextID
	ls.lanes[lane.id] = lane
	ls.order = append(ls.order, lane.id)
	ls.liveLanes.Store(int64(len(ls.lanes)))
	return inference.LaneHandle{ID: lane.id}, nil
}

// laneOverlapPrefillChunkRows caps an overlapped admission's prefill chunk
// width (rows per command buffer) while other lanes are streaming. 512 rows ≈
// one 26B chunk of ~0.35s — the in-flight streams' worst inter-token gap —
// versus ~1.2s for the whole-prompt buffer; the joiner's own prefill pays the
// per-chunk seams (~6% by the #367 depth-ladder receipts), a TTFT tax of tens
// of milliseconds on a joiner that is by definition not the only request.
const laneOverlapPrefillChunkRows = 512

// prefillLane runs the lane's prompt through the session's PRODUCTION prefill
// route (prefillRetainedTokens: chunked batched-dense forward, flash prompt
// SDPA, the kv-share skip — everything the plain serve path prefills with)
// and takes the boundary hidden as the lane's decode seed. This replaced the
// original one-token-at-a-time stepBody loop: byte-compatible (the sampled
// and E2B oracle receipts compare lanes against classic sessions prefilled by
// exactly this route) but ~40× cheaper on admission — the serial loop paid
// one commit+wait per prompt token (measured live: 4 concurrent
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
		// PART A — RAGGED harvest for free-running chained lanes: poll each
		// lane's in-flight round and, the instant one completes, read its
		// 4-byte token, commit-or-drop the pre-encoded held round (commit is
		// µs — the lane rolls on without waiting any other lane or the Step
		// boundary; a terminal token drops the held round un-committed: zero
		// waste, no rewind), then pre-encode the next held round overlapped
		// with the new in-flight one. The Step returns whichever lanes
		// completed — the slowest lane no longer gates the batch.
		var pipeline, classic []*decodeLane
		for _, lane := range active {
			if lane.inFlightLive {
				pipeline = append(pipeline, lane)
			} else {
				classic = append(classic, lane)
			}
		}
		harvest := func(lane *decodeLane) bool {
			if lane.inFlightCB.Status() < metal.MTLCommandBufferStatusCompleted {
				return false
			}
			tok := lane.inFlightScr.token()
			terminal := tok < 0 || int(tok) >= ls.vocab || lane.stops[tok] || lane.generated+1 >= lane.maxNew
			// Commit-or-drop the held round FIRST (µs) so the lane's GPU work
			// resumes before any host bookkeeping.
			if terminal {
				ls.dropHeldRound(lane)
			} else if lane.heldLive {
				ls.commitChainedRound(lane, lane.heldCB)
			}
			waitReleaseChainedCB(lane.inFlightCB) // complete — returns immediately; span instrument
			lane.sess.headEnc.putGreedyScratch(lane.inFlightScr)
			lane.inFlightCB, lane.inFlightScr, lane.inFlightLive = metal.MTLCommandBufferObject{}, nil, false
			if tok < 0 || int(tok) >= ls.vocab {
				stepErr = core.NewError("native.laneSet.Step: chained head returned an invalid token")
				return true
			}
			lane.generated++
			results = append(results, inference.LaneStep{
				Lane:     inference.LaneHandle{ID: lane.id},
				Token:    tok,
				HasToken: true,
				Terminal: terminal,
			})
			if terminal {
				lane.terminal = true
				lane.pos = lane.sess.pos
				return true
			}
			if lane.heldLive {
				// The held round became the in-flight round at the commit above;
				// pre-encode the next one overlapped with its execution.
				lane.inFlightCB, lane.inFlightScr, lane.inFlightLive = lane.heldCB, lane.heldScr, true
				lane.heldCB, lane.heldScr, lane.heldLive = metal.MTLCommandBufferObject{}, nil, false
				if herr := ls.holdNextRound(lane); herr != nil {
					stepErr = herr
					return true
				}
			} else if lane.chainDead {
				// Mid-stream demote (defensive — a head that chained at entry
				// declines mid-stream only under test hooks): the emitted token
				// is not yet advanced; Phase 2 advances it per-lane next Step.
				lane.pendingToken = tok
				lane.needAdvance = true
			}
			lane.pos = lane.sess.pos
			return true
		}
		if len(pipeline) > 0 {
			progress := false
			for _, lane := range pipeline {
				if harvest(lane) {
					progress = true
				}
				if stepErr != nil {
					return
				}
			}
			// Block until at least one lane completes when nothing else will
			// produce a token this Step — the Step contract stays "progress or
			// the set is done". A single in-flight lane blocks directly on its
			// cb (polling only pays when there are lanes to rag between).
			for !progress && len(classic) == 0 {
				if err := ctx.Err(); err != nil {
					stepErr = err
					return
				}
				live := 0
				var sole *decodeLane
				for _, lane := range pipeline {
					if lane.inFlightLive {
						live++
						sole = lane
					}
				}
				if live == 0 {
					break
				}
				if live == 1 {
					waitUntilCompletedFast(sole.inFlightCB) // harvest re-checks status and releases
				} else {
					lanePollWait()
				}
				for _, lane := range pipeline {
					if lane.inFlightLive && harvest(lane) {
						progress = true
					}
					if stepErr != nil {
						return
					}
				}
			}
			if progress {
				ls.fwdCount++
				ls.chainedSteps++
			}
		}

		// PART B — entry/classic lanes: one token each from the lane's
		// current hidden — the lane's own sampler for a non-greedy
		// discipline, else head+greedy, with an ALL-GREEDY set taking ONE
		// batched head submission (the MTP verify's K-row fused
		// lm_head+argmax) instead of K commit+waits. A needAdvance lane
		// (chain exit with its token already emitted) skips the emit and
		// goes straight to Phase 2's advance.
		var emitLanes []*decodeLane
		for _, lane := range classic {
			if lane.needAdvance {
				lane.needAdvance = false
				advancing = append(advancing, lane)
				continue
			}
			emitLanes = append(emitLanes, lane)
		}
		classic = emitLanes
		batchedToks, batchedHead := ls.phase1GreedyRows(classic)
		var sampledToks []int32
		batchedSampled := false
		if !batchedHead {
			var serr error
			sampledToks, batchedSampled, serr = ls.phase1SampledTopKRows(classic)
			if serr != nil {
				stepErr = serr
				return
			}
			if !batchedSampled {
				sampledToks, batchedSampled, serr = ls.phase1SampledLogitsRows(classic)
				if serr != nil {
					stepErr = serr
					return
				}
			}
		}
		hi := 0
		for _, lane := range classic {
			var tok int32
			var err error
			switch {
			case batchedHead:
				tok = batchedToks[hi]
				hi++
			case batchedSampled:
				tok = sampledToks[hi]
				hi++
				if lane.sampleParams.RepeatPenalty > 1 {
					lane.sampleHistory = append(lane.sampleHistory, tok)
				}
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
		// one-encoder batch); greedy re-encode lanes CHAIN (the forward carries
		// the head argmax + next-embed in the same cb — the serial chained
		// decode's shape, per lane); the remaining re-encode lanes advance
		// through the plain per-token machinery, grouped when shared-encode
		// eligible. One model per set makes the partition near-uniform in
		// practice; the split keeps every mix correct.
		var replay, reenc, chain []*decodeLane
		for _, lane := range advancing {
			switch {
			case !lane.reencode:
				replay = append(replay, lane)
			case ls.laneChainable(lane):
				chain = append(chain, lane)
			default:
				reenc = append(reenc, lane)
			}
		}
		if len(chain) > 0 {
			if err := ls.chainEnterForward(chain); err != nil {
				stepErr = err
				return
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

// cbChainDisabled forces every lane onto the two-phase path (batched Phase-1
// heads + rendezvous'd Phase-2) — LTHN_CB_CHAIN=0, the A/B lever for the
// chained-free-run-vs-batched-tail economics (the K≥8 tail re-check). Same
// receipted-off switch convention as LTHN_CB_STEP / LTHN_CB_GEMM.
var cbChainDisabled = os.Getenv("LTHN_CB_CHAIN") == "0"

// laneChainable reports whether a re-encode lane can run the CHAINED step:
// greedy discipline (the chain's head is a GPU argmax), a shared-encode
// eligible session (the chain encodes into a grouped submission), the GPU
// next-inputs seam present (the chain's embed producer), and the head not
// having declined before. Sampled lanes and declined lanes keep the
// two-phase path, as does the whole set under LTHN_CB_CHAIN=0.
func (ls *laneSet) laneChainable(lane *decodeLane) bool {
	return !cbChainDisabled && lane.sampler == nil && !lane.chainDead &&
		lane.sess.sharedStepEligible() &&
		lane.sess.encNextInputsGPU != nil && lane.sess.plScratchNew != nil &&
		lane.sess.headEnc != nil
}

// chainedForward advances the chained lanes through one grouped submission
// round in the serial chained-live decode's per-lane shape: each lane's cb
// carries [forward → head argmax → next-token embed gather into xA], all K
// committed back-to-back, waits pipelined. After the waits a lane holds its
// NEXT token (chainTok, a 4-byte readback) and its next input already on-GPU
// — the following Step emits the token with no head submission, no hidden
// readback and no host embed. A lane whose head declines mid-encode is
// DEMOTED (chainDead): its forward is still valid, so it commits and reads
// the hidden classically, staying two-phase from then on. A session-encode
// error abandons that lane's uncommitted cb and retries it per-lane —
// encode-time work mutates no session state.
// chainRoundCommit encodes ONE chained round for the lane — [forward → head
// argmax → next-token embed gather into xA] — and COMMITS it without
// waiting. fromXA binds the input the previous round's tail gathered into xA
// on-GPU; the entry round embeds lane.pendingToken on host instead. The
// chainRoundEncode encodes ONE chained round for the lane — [forward → head
// argmax → next-token embed gather into xA] — and returns it PRE-ENCODED,
// retained, UNCOMMITTED: the caller commits it (µs) the instant the previous
// round completes, so the encode cost always overlaps an executing round and
// a terminal token just drops the held cb (no advance happened — no rewind,
// no truncation). fromXA binds the input the previous round's tail gathers
// into xA at execution; the entry round embeds lane.pendingToken on host.
// ok=false means the head declined (no side effects — nothing committed):
// the lane leaves the chained path. The session position is NOT advanced
// here; commitChainedRound bumps it.
func (ls *laneSet) chainRoundEncode(lane *decodeLane, fromXA bool) (metal.MTLCommandBufferObject, *headGreedyScratch, bool, error) {
	var none metal.MTLCommandBufferObject
	sink := &sharedStepSink{cb: commandBufferFast(queue)}
	sink.enc = computeCommandEncoderFast(sink.cb)
	if lane.chainSc == nil {
		lane.chainSc = lane.sess.plScratchNew()
	}
	var err error
	if fromXA {
		err = lane.sess.stepEncodeSharedChained(sink)
	} else {
		err = lane.sess.stepIDEncodeShared(lane.pendingToken, sink)
	}
	if err != nil {
		return none, nil, false, err
	}
	scr, gok, gerr := lane.sess.headEnc.encodeGreedy(sink.enc, sink.finalOut, nil)
	if gerr != nil {
		endEncodingFast(sink.enc)
		return none, nil, false, gerr
	}
	if !gok {
		endEncodingFast(sink.enc)
		return none, nil, false, nil
	}
	if nerr := lane.sess.encNextInputsGPU(sink.enc, scr.outToken, lane.sess.state.xA, lane.chainSc); nerr != nil {
		lane.sess.headEnc.putGreedyScratch(scr)
		endEncodingFast(sink.enc)
		return none, nil, false, nerr
	}
	endEncodingFast(sink.enc)
	// The round crosses the Step autorelease-pool boundary: pin it (the pool
	// drains at Step end and would free the autoreleased cb — a later wait
	// then hangs forever, the stepGreedyLiveCommit no-pool trap).
	sink.cb.Retain()
	return sink.cb, scr, true, nil
}

// commitChainedRound submits a pre-encoded round and advances the session
// position — the µs step that keeps a lane rolling the instant its previous
// round completes.
func (ls *laneSet) commitChainedRound(lane *decodeLane, cb metal.MTLCommandBufferObject) {
	commitCommandBufferFast(cb)
	lane.sess.pos++
}

// waitReleaseChainedCB completes a retained, COMMITTED chained round's
// command buffer and drops the cross-pool pin chainRoundEncode took. Never
// call it on a held (uncommitted) round — the wait would hang forever;
// dropHeldRound is that path.
func waitReleaseChainedCB(cb metal.MTLCommandBufferObject) {
	waitUntilCompletedFast(cb)
	if pieceTimingOn {
		laneGPUSpanNs += int64(float64(cb.GPUEndTime()-cb.GPUStartTime()) * 1e9)
	}
	cb.Release()
}

// lanePollWait is the ragged harvest's completion-poll interval: rounds run
// milliseconds, so a 50µs sleep costs <1%% of one core while keeping harvest
// latency negligible against round time.
func lanePollWait() { time.Sleep(50 * time.Microsecond) }

// dropHeldRound discards a lane's pre-encoded, never-committed round — the
// terminal path. Nothing executed, nothing advanced: no rewind, no KV
// truncation, just the pin and the scratch.
func (ls *laneSet) dropHeldRound(lane *decodeLane) {
	if !lane.heldLive {
		return
	}
	lane.heldCB.Release()
	lane.sess.headEnc.putGreedyScratch(lane.heldScr)
	lane.heldCB, lane.heldScr, lane.heldLive = metal.MTLCommandBufferObject{}, nil, false
}

// holdNextRound pre-encodes the lane's next chained round when the budget
// still has room for its emission (in-flight emits generated+1; the held
// round emits generated+2). A head decline demotes the lane chainDead with
// no side effects.
func (ls *laneSet) holdNextRound(lane *decodeLane) error {
	if lane.chainDead || lane.generated+2 > lane.maxNew {
		return nil
	}
	cb, scr, ok, err := ls.chainRoundEncode(lane, true)
	if err != nil {
		return err
	}
	if !ok {
		lane.chainDead = true
		return nil
	}
	lane.heldCB, lane.heldScr, lane.heldLive = cb, scr, true
	return nil
}

// chainEnterForward starts each chainable lane's free-running pipeline: the
// entry round (host-embed) is encoded and committed immediately (inFlight),
// and the next round is pre-encoded and HELD uncommitted behind it — from
// here the lane self-sustains: each harvest commits the held round and
// pre-encodes another. A decline or encode error before the entry commit has
// no side effects; the lane advances per-lane instead, demoted.
func (ls *laneSet) chainEnterForward(chain []*decodeLane) error {
	entered := false
	for _, lane := range chain {
		cb, scr, ok, err := ls.chainRoundEncode(lane, false)
		if err != nil || !ok {
			lane.chainDead = true
			h, serr := lane.sess.stepIDInPool(lane.pendingToken)
			if serr != nil {
				return serr
			}
			lane.hidden = append(lane.hidden[:0], h...)
			lane.pos = lane.sess.pos
			continue
		}
		ls.commitChainedRound(lane, cb)
		lane.inFlightCB, lane.inFlightScr, lane.inFlightLive = cb, scr, true
		lane.pos = lane.sess.pos
		entered = true
		if herr := ls.holdNextRound(lane); herr != nil {
			return herr
		}
	}
	if entered {
		ls.fwdCount++
		ls.chainedSteps++
	}
	return nil
}

// drainLanePending completes and discards a lane's chained rounds — the
// Retire/Close/error path for a lane leaving mid-pipeline: the committed
// in-flight round must finish before its session closes; the held round was
// never committed and is simply dropped.
func (ls *laneSet) drainLanePending(lane *decodeLane) {
	if lane == nil {
		return
	}
	if lane.inFlightLive {
		waitReleaseChainedCB(lane.inFlightCB)
		lane.sess.headEnc.putGreedyScratch(lane.inFlightScr)
		lane.inFlightCB, lane.inFlightScr, lane.inFlightLive = metal.MTLCommandBufferObject{}, nil, false
	}
	ls.dropHeldRound(lane)
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
	ls.liveLanes.Store(int64(len(ls.lanes)))
	for i, id := range ls.order {
		if id == h.ID {
			ls.order = append(ls.order[:i], ls.order[i+1:]...)
			break
		}
	}
	ls.drainLanePending(lane) // an in-flight chained round must complete before its session closes
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
		ls.drainLanePending(lane)
		if err := lane.sess.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	ls.gemm.release()
	ls.gemm = nil
	ls.lanes = nil
	ls.order = nil
	ls.liveLanes.Store(0)
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
