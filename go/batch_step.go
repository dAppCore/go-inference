// SPDX-Licence-Identifier: EUPL-1.2

package inference

import "context"

// BatchStepModel is the OPTIONAL capability of a model whose engine can advance
// several INDEPENDENT decode sessions ("lanes") by one token each through a
// single batched forward — the multi-session continuous-batching step. Each
// lane owns its own KV cache, scalar position, and sampler state; the lanes
// share the model's immutable weights. It is probed exactly as SchedulerModel
// and TokenizerModel are: present and available ⇒ the scheduler's interleave
// mode may drive a shared step coordinator; absent or unavailable ⇒ the
// scheduler keeps its existing per-request path unchanged, byte-for-byte.
//
// The kill switch LTHN_CB_STEP=0 makes a capable engine report the capability
// UNAVAILABLE (OpenLaneSet refuses with a clear error rather than silently
// degrading to a serial loop), so serial mode and interleave-without-CB stay
// byte-identical to a build that never had the capability. Availability is a
// method rather than a wrapper type deliberately: wrapping the model to
// add/remove the interface would strip its other optional capabilities
// (TokenizerModel, CancellableModel, …) the way the registry capability-
// stripping bug class did — a probe-then-check keeps every other capability
// reachable on the same value.
type BatchStepModel interface {
	// BatchStepAvailable reports whether OpenLaneSet will build a real
	// multi-session owner right now. False when the kill switch is set
	// (LTHN_CB_STEP=0) or the loaded engine/arch cannot bind a lane set.
	BatchStepAvailable() bool
	// OpenLaneSet opens a multi-session owner over the model's shared weights.
	// Returns a clear error (never a serial-loop fallback) when the capability
	// is unavailable — the caller must treat that as "no CB stepping" and stay
	// on its existing path.
	OpenLaneSet(cfg LaneSetConfig) (LaneSet, error)
}

// LaneSetConfig bounds a lane set at construction.
type LaneSetConfig struct {
	// MaxLanes caps concurrent lanes — the KV co-residency budget. Zero or
	// negative means the engine picks a conservative default.
	MaxLanes int
}

// LaneHandle identifies one admitted lane within a LaneSet. The zero value is
// never a valid handle (lane ids start at 1), so a zero LaneHandle reads as
// "no lane".
type LaneHandle struct {
	ID int
}

// Valid reports whether h names an admitted lane (ids start at 1).
func (h LaneHandle) Valid() bool { return h.ID > 0 }

// LaneSpec admits one lane: its prompt token ids and per-lane decode params.
type LaneSpec struct {
	// PromptIDs is the lane's prompt, already tokenised. Prefilled at admission.
	PromptIDs []int32
	// MaxNew caps the lane's generated tokens; the lane goes Terminal after it.
	MaxNew int
	// StopTokens end the lane when produced (in addition to the model's own).
	StopTokens []int32
	// Sampler selects the decode discipline. The zero value is greedy; a
	// non-greedy config (temperature, min-p, or repeat-penalty engaged — the
	// same pick the plain serve path makes) gives the lane its OWN sampler
	// state, token-identical to the plain sampled generate at the same seed.
	Sampler SamplerConfig
	// SampleSeed seeds the lane's sampler RNG stream when Sampler selects a
	// non-greedy discipline. Greedy lanes ignore it. The zero seed is a valid
	// seed, mirroring GenerateConfig.Seed semantics.
	SampleSeed uint64
}

// LaneStep is one lane's result from a single batched Step.
type LaneStep struct {
	// Lane is the lane this result belongs to.
	Lane LaneHandle
	// Token is the token this lane produced this step (valid when HasToken).
	Token int32
	// HasToken is false when the lane produced no token this step — it was
	// already terminal, or terminated on this token with nothing to emit.
	HasToken bool
	// Terminal marks a lane that hit a stop token or MaxNew this step. The
	// caller should Retire it; a terminal lane is not advanced by later Steps.
	Terminal bool
}

// LaneSet is the multi-session owner: K lanes that share the model's immutable
// weights, each owning its own KV cache, position, and sampler state, advanced
// by ONE batched forward per Step. Ragged admission is supported — Prepare may
// be called between Steps to join a new lane to the running set.
//
// A LaneSet is single-goroutine per instance (the lanes' decode state is
// mutated without synchronisation, exactly as a single engine Session is): the
// step coordinator that owns it drives Prepare/Step/Retire from one goroutine.
type LaneSet interface {
	// Prepare admits a new lane, prefilling its prompt, and returns its handle.
	// Safe to call between Steps (ragged admission). Errors if the lane set is
	// at MaxLanes or the spec is unserviceable (e.g. non-greedy Sampler on a
	// greedy-only owner).
	Prepare(ctx context.Context, spec LaneSpec) (LaneHandle, error)
	// Step advances every active, non-terminal lane by one token in a single
	// batched forward and returns one LaneStep per lane that was advanced.
	// Returns an empty slice (no error) when no lane is active.
	Step(ctx context.Context) ([]LaneStep, error)
	// Retire removes a lane and releases its per-lane state. The shared weights
	// stay resident (they belong to the model, not the lane).
	Retire(h LaneHandle) error
	// Active reports the number of admitted, non-retired lanes.
	Active() int
	// BatchForwardCount is the monotonic count of batched forwards executed by
	// Step. A conformance fixture asserts it advanced by exactly the number of
	// Steps taken — proof the K-way path fired, which a fake that looped K
	// serial single-session steps could never provide.
	BatchForwardCount() uint64
	// Close retires every remaining lane and releases the owner.
	Close() error
}
