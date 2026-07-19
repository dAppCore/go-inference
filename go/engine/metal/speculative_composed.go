// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// speculative_composed.go is the composed-family binding of the ONE speculative route:
// LoadSpeculativePair dispatches here when the TARGET checkpoint is a composed/hybrid arch (a Qwen
// 3.5/3.6 base cannot be an ArchSession — its gated-delta layers thread recurrent state, not KV the
// gemma4 AssistantPair machinery could share). The pairing itself lives model-side
// (composed.SpeculativePair — trained-shape drafter feed, per-token byte-exact verify, the measured
// block-verify lane); this file only adapts it to the same serve surfaces the gemma4 pair exposes:
// the shared engine.TextModel wrap (templates, stops, sampling, vision refusals — all inherited from
// the plain composed serve source) plus inference.SpeculativeMetricsProvider for the -draft bench
// read. Greedy requests run the pair; sampled requests keep the plain composed lane (the pair's v1
// scope — the drafter only ever decides how many of the base's own greedy tokens commit per round).
package native

import (
	"os"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/engine"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
)

// mtpBlockVerifyEnabled arms the composed pair's block-verify lane (one batched base forward per
// round). OFF by default on the measured evidence (dev@82d263d: 48 tokens, per-token 15.7s vs block
// 21.1s on the 27B — a 4-row batched forward costs ~3x a single step on the host lane, so the
// forward-count win inverts), and token-identity tier besides. The lever exists so the lane can be
// re-measured as its economics change (KV truncate-on-restore, #8-B quant tails, an MoE base).
var mtpBlockVerifyEnabled = os.Getenv("LTHN_MTP_BLOCK") == "1"

// loadComposedSpeculativePair loads a composed base + its MTP drafter as one speculative
// inference.TextModel — the composed-family arm of LoadSpeculativePair (the serving.SpeculativeLoader
// shape). The pair validates the attachment from the drafter's DECLARATION before any base tensor is
// realised (composed.LoadSpeculativePairDirs); the serve surfaces come from the same engine.TextModel
// wrap the plain composed path uses, so a paired serve frames ChatML, honours stops and declines
// vision exactly as an unpaired one.
func loadComposedSpeculativePair(targetPath, draftPath string, draftBlock int, opts ...inference.LoadOption) (inference.TextModel, error) {
	cfg := inference.ApplyLoadOpts(opts)
	pair, err := composed.LoadSpeculativePairDirs(targetPath, draftPath)
	if err != nil {
		return nil, core.E("native.loadComposedSpeculativePair", "load pair", err)
	}
	pair.BlockVerify = mtpBlockVerifyEnabled
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(targetPath, "tokenizer.json"))
	if err != nil {
		_ = pair.Close()
		return nil, core.E("native.loadComposedSpeculativePair", "load tokenizer", err)
	}
	modelType := probeModelType(targetPath)
	serveLen := cfg.ContextLen
	if serveLen <= 0 {
		serveLen = resolveDefaultContext(model.ProbeDirContextWindow(targetPath))
	}
	tm := composed.NewTokenModel(pair.Base)
	info := inference.ModelInfo{
		Architecture: modelType,
		VocabSize:    tm.Vocab(),
		NumLayers:    tm.NumLayers(),
		HiddenSize:   tm.HiddenSize(),
	}
	rec := &composedSpecMetrics{}
	src := &composedSpeculativeTextModel{
		sessionTextModel: sessionTextModel{sm: tm, tok: tok, modelType: modelType, numLayers: tm.NumLayers()},
		pair:             pair,
		rec:              rec,
		draftBlock:       draftBlock,
	}
	return &composedSpeculativePairModel{
		TextModel: engine.NewTextModel(src, tok, modelType, info, serveLen),
		pair:      pair,
		rec:       rec,
	}, nil
}

// composedSpecMetrics shares the last generation's speculative counters between the engine session
// (which records them) and the pair TextModel (which serves SpeculativeMetricsProvider).
type composedSpecMetrics struct {
	mu   sync.Mutex
	last inference.SpeculativeMetrics
}

func (r *composedSpecMetrics) record(m composed.SpeculativeMetrics, emitted int, wall time.Duration) {
	sm := inference.SpeculativeMetrics{
		ProposedTokens:    m.ProposedTokens,
		AcceptedTokens:    m.AcceptedTokens,
		RejectedTokens:    m.RejectedTokens,
		TargetVerifyCalls: m.TargetVerifyCalls,
		TargetCalls:       m.TargetVerifyCalls,
		DraftCalls:        m.DraftCalls,
		AcceptanceRate:    m.AcceptanceRate,
		WallDuration:      wall,
	}
	if wall > 0 {
		sm.VisibleTokensPerSec = float64(emitted) / wall.Seconds()
	}
	r.mu.Lock()
	r.last = sm
	r.mu.Unlock()
}

func (r *composedSpecMetrics) snapshot() inference.SpeculativeMetrics {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.last
}

// composedSpeculativeTextModel is the engine.TokenModel source for a paired composed serve: the plain
// composed source (templates, vision declarations, Close) with sessions that route greedy generation
// through the pair.
type composedSpeculativeTextModel struct {
	sessionTextModel
	pair       *composed.SpeculativePair
	rec        *composedSpecMetrics
	draftBlock int
}

func (m *composedSpeculativeTextModel) OpenEngineSession() (engine.Session, error) {
	plain, err := m.sessionTextModel.OpenEngineSession()
	if err != nil {
		return nil, err
	}
	sess, ok := plain.(*composedEngineSession)
	if !ok {
		return nil, core.NewError("native.composedSpeculativeTextModel: plain source returned an unexpected session type")
	}
	return &composedSpeculativeEngineSession{
		composedEngineSession: sess,
		pair:                  m.pair,
		rec:                   m.rec,
		draftBlock:            m.draftBlock,
	}, nil
}

// composedSpeculativeEngineSession routes the GREEDY generate through the MTP pair; everything else —
// sampled generation, multimodal turns (spliced rows are outside the pair's v1 scope), continuity
// appends, -state capture — is the embedded plain session, unchanged.
type composedSpeculativeEngineSession struct {
	*composedEngineSession
	pair       *composed.SpeculativePair
	rec        *composedSpecMetrics
	draftBlock int
}

func (s *composedSpeculativeEngineSession) GenerateFromCacheEach(maxNew, eosID int, yield func(int32) bool) ([]int32, error) {
	if len(s.embRows) > 0 {
		// A multimodal turn replays spliced embedding rows the pair does not consume — plain lane.
		return s.composedEngineSession.GenerateFromCacheEach(maxNew, eosID, yield)
	}
	var out []int32
	t0 := time.Now()
	m, err := s.pair.GenerateSpeculativeEach(s.prompt, maxNew, eosID, s.draftBlock, func(id int32) bool {
		out = append(out, id)
		return yield(id)
	})
	s.rec.record(m, len(out), time.Since(t0))
	if mtpDiagForTest {
		nativeTraceLog(core.Sprintf("mtp-diag composed pair: emitted=%d proposed=%d accepted=%d acceptance=%.1f%% draftCalls=%d baseForwards=%d block=%v\n",
			len(out), m.ProposedTokens, m.AcceptedTokens, m.AcceptanceRate*100, m.DraftCalls, m.TargetVerifyCalls, s.pair.BlockVerify))
	}
	return out, err
}

// composedSpeculativePairModel is the served pair: the shared engine.TextModel wrap plus the
// speculative-metrics capability and the pair's own close (the base checkpoint mapping).
type composedSpeculativePairModel struct {
	inference.TextModel
	pair *composed.SpeculativePair
	rec  *composedSpecMetrics
}

var (
	_ inference.TextModel                  = (*composedSpeculativePairModel)(nil)
	_ inference.SpeculativeMetricsProvider = (*composedSpeculativePairModel)(nil)
	_ inference.SerialModel                = (*composedSpeculativePairModel)(nil)
)

// SpeculativeMetrics reports the last generation's draft/verify counters (the -draft bench read).
func (m *composedSpeculativePairModel) SpeculativeMetrics() inference.SpeculativeMetrics {
	return m.rec.snapshot()
}

// SerialGeneration declares the pair single-session: the drafter's persistent head session is a
// shared singleton (reset per generation), so concurrent generations would race it — the scheduler
// serialises this model's lane, exactly as the gemma4 pair declares.
func (m *composedSpeculativePairModel) SerialGeneration() bool { return true }

// Close releases the engine wrap and the pair's base checkpoint mapping.
func (m *composedSpeculativePairModel) Close() core.Result {
	r := m.TextModel.Close()
	if err := m.pair.Close(); err != nil {
		return core.Fail(core.E("native.composedSpeculativePairModel", "close pair", err))
	}
	return r
}
