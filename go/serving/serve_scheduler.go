// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/serving/scheduler"
)

// parseSchedulerMode maps the -scheduler flag to a scheduler.Mode, failing
// closed on any value the scheduler does not implement (so a typo boots to an
// error, never a silent default).
func parseSchedulerMode(flag string) (scheduler.Mode, error) {
	switch scheduler.Mode(core.Lower(core.Trim(flag))) {
	case scheduler.ModeSerial:
		return scheduler.ModeSerial, nil
	case scheduler.ModeBatch:
		return scheduler.ModeBatch, nil
	case scheduler.ModeInterleave:
		return scheduler.ModeInterleave, nil
	default:
		return "", core.E("serving.scheduler", "unknown -scheduler mode "+flag+" (want serial|batch|interleave)", nil)
	}
}

// schedulerServeConfig maps a ServeConfig into the scheduler.Config the serve
// path builds around each model. Sizings are single-model serve defaults; the
// mode comes from the -scheduler flag (set by the caller). The model's own
// tokeniser supplies the prompt-token count for the batch/interleave budget, so
// no byte-cost model is fabricated here.
func schedulerServeConfig(mode scheduler.Mode, concurrency int) scheduler.Config {
	if concurrency <= 0 {
		concurrency = 4
	}
	return scheduler.Config{
		Mode:            mode,
		MaxConcurrent:   concurrency,
		MaxQueue:        64,
		StreamBuffer:    16,
		MaxBatchTokens:  0, // count-only by default; a token budget is a future flag
		RequestIDPrefix: "serve",
	}
}

// schedulerResolver decorates a Resolver so every resolved model is presented
// to the mux through a persistent scheduler.Model in the configured mode. The
// mux's request path already prefers inference.SchedulerModel.Schedule when the
// resolved model implements it (serving/compat/mux.go's forEachCompatToken), so
// wrapping here is all that is needed to route requests between the HTTP
// handlers and the model through the scheduler — no change to the mux itself.
//
// One scheduler is built per underlying model and reused for every resolve of
// that same model, so concurrent requests share its queue / running-set (the
// whole point of a scheduler). A hot-swap that returns a different model builds
// a fresh scheduler and tears the previous engine down; the underlying model's
// lifecycle stays with the resolver beneath (CloseEngine, not Close).
type schedulerResolver struct {
	base Resolver
	cfg  scheduler.Config

	mu        sync.Mutex
	lastModel inference.TextModel
	sched     *scheduler.Model
}

func newSchedulerResolver(base Resolver, cfg scheduler.Config) *schedulerResolver {
	return &schedulerResolver{base: base, cfg: cfg}
}

// ResolveModel resolves through the base resolver, then returns a scheduler
// wrapping the resolved model — building (or rebuilding, on a model swap) the
// scheduler lazily. A capability the mode requires that the model lacks fails
// here (scheduler.New's fail-closed probe), surfacing to the caller as a
// resolve error rather than a silent downgrade.
func (r *schedulerResolver) ResolveModel(ctx context.Context, name string) (inference.TextModel, error) {
	model, err := r.base.ResolveModel(ctx, name)
	if err != nil {
		return nil, err
	}
	if model == nil {
		return nil, core.E("serving.scheduler", "resolver returned a nil model", nil)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.sched != nil && r.lastModel == model {
		return r.sched, nil
	}
	sched, err := scheduler.New(model, r.cfg)
	if err != nil {
		return nil, core.E("serving.scheduler", "cannot wrap the resolved model in "+string(r.cfg.Mode)+" mode", err)
	}
	if r.sched != nil {
		r.sched.CloseEngine() // tear down the previous engine; the old model belongs to the base resolver
	}
	r.sched, r.lastModel = sched, model
	return sched, nil
}

// current returns the scheduler the resolver has most recently built (nil until
// the first resolve) — the observation seam serve wiring tests read Stats from.
func (r *schedulerResolver) current() *scheduler.Model {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.sched
}

// close tears down the active scheduler's engine goroutines. The wrapped model
// is left to the base resolver, which owns its lifecycle.
func (r *schedulerResolver) close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.sched != nil {
		r.sched.CloseEngine()
	}
}
