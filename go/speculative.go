// SPDX-Licence-Identifier: EUPL-1.2

package inference

import "time"

// SpeculativeMetrics reports one speculative-decode operation: a drafter
// proposes token runs the target model verifies in single forward passes.
// Engine-neutral — the Metal engine's gemma4 MTP drafter populates it today;
// hip/cuda speculative lanes share the shape. Zero-valued when the last
// operation ran the plain autoregressive path.
//
//	if p, ok := m.(inference.SpeculativeMetricsProvider); ok {
//	    sm := p.SpeculativeMetrics()
//	    fmt.Printf("accept %.0f%% over %d rounds\n", sm.AcceptanceRate*100, sm.TargetVerifyCalls)
//	}
type SpeculativeMetrics struct {
	// Draft/verify counters
	DraftTokenSchedule []int // per-round draft lengths actually proposed
	ProposedTokens     int   // drafter tokens offered for verification
	AcceptedTokens     int   // drafter tokens the target accepted
	RejectedTokens     int   // drafter tokens the target rejected
	TargetVerifyCalls  int   // target forward passes spent verifying drafts
	TargetCalls        int   // total target forward passes (verify + fallback)
	DraftCalls         int   // drafter forward passes

	// Rates (computed)
	AcceptanceRate         float64 // AcceptedTokens / ProposedTokens
	VisibleTokensPerSec    float64 // emitted tokens / WallDuration
	TargetTokensPerSec     float64 // target-equivalent throughput
	WarmDecodeTokensPerSec float64 // steady-state decode rate excluding warmup

	// Timing
	WallDuration         time.Duration // whole speculative operation
	RestoreDuration      time.Duration // KV rollback after rejections
	TargetVerifyDuration time.Duration // time inside target verify passes
	TargetDuration       time.Duration // time inside all target passes
	DraftDuration        time.Duration // time inside drafter passes

	// Memory
	PeakMemoryBytes uint64 // peak device memory during the operation
}

// SpeculativeMetricsProvider is the optional capability a [TextModel] exposes
// when it ran (or can run) a speculative-decode lane. Probe it the same way
// as [AttentionInspector]:
//
//	if p, ok := model.(inference.SpeculativeMetricsProvider); ok {
//	    sm := p.SpeculativeMetrics()
//	    _ = sm
//	}
type SpeculativeMetricsProvider interface {
	// SpeculativeMetrics returns counters from the last completed operation.
	// Zero-valued (ProposedTokens == 0) when speculation did not engage.
	SpeculativeMetrics() SpeculativeMetrics
}

// SpeculativePairBackend is the optional [Backend] capability for an engine
// that can load a target checkpoint + drafter as one speculative-decode
// [TextModel] (MTP draft→verify, or an equivalent method the drafter
// declares). It is probed on the REGISTERED BACKEND rather than on a loaded
// model — unlike [AttentionInspector] / [VisionModel] (capabilities of a
// TextModel that is already loaded), loading THE PAIR is the operation being
// asked for, so there is no model yet to probe:
//
//	b, ok := inference.Get("metal")
//	if !ok { return core.NewError("metal backend not registered") }
//	spl, ok := b.(inference.SpeculativePairBackend)
//	if !ok { return core.NewError("metal backend exposes no speculative-pair loader") }
//	tm, err := spl.LoadSpeculativePair(targetPath, draftPath, 5)
//
// The metal engine implements this by delegating to its existing pair-loading
// machinery (assistant-pair + composed-pair loading, already proven by serve's
// MTP lane); dappco.re/go/inference/train/tune discovers it this way — via
// [Default] or [Get] plus a type assertion, exactly how [LoadModel] discovers a
// plain-load [Backend] — to run the MTP draft-block autotune sweep without
// importing any concrete engine.
type SpeculativePairBackend interface {
	// LoadSpeculativePair loads targetPath + draftPath as one speculative
	// TextModel running draftBlock-wide MTP verify forwards. draftBlock <= 0
	// defers to the engine's own default.
	LoadSpeculativePair(targetPath, draftPath string, draftBlock int, opts ...LoadOption) (TextModel, error)
}
