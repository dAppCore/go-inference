// SPDX-Licence-Identifier: EUPL-1.2

// Package weightsync coordinates pushing a new weight version into a running
// model and swapping it in atomically — the online / reinforcement-learning
// update path, where a freshly trained checkpoint is handed to a live serving
// engine without a restart.
//
// The mechanism mirrors an online weight-update / checkpoint-engine flow: a new
// version is first staged into a buffer, then atomically activated (staged →
// live). The actual GPU weight apply is supplied by the caller through the
// Applier interface; this package owns only the versioning and the swap
// coordination, so it stays pure-Go and trivially testable.
//
//	co := weightsync.New(myApplier)
//	co.Update(ctx, 1, "ckpt-iter-1")        // stage then activate
//	live := co.Current()                    // 1
//
// A version must strictly advance: re-applying the live version, or an older
// one, is rejected with ErrStaleVersion so a duplicate or out-of-order push can
// never roll the engine backwards.
package weightsync

import (
	"context"
	"sync"
	"time"

	core "dappco.re/go"
)

// Sentinel errors. Callers match them with core.Is so the failure reason is
// inspectable regardless of the operation context wrapped around them.
//
//	if core.Is(err, weightsync.ErrStaleVersion) { skipDuplicate() }
var (
	// ErrStaleVersion is returned when an Update targets a version less than or
	// equal to the live version (a duplicate or out-of-order push).
	ErrStaleVersion = core.NewError("weightsync: version not newer than current")
	// ErrStageFailed wraps an Applier.Stage failure; the live version is left
	// unchanged and nothing is marked pending.
	ErrStageFailed = core.NewError("weightsync: stage failed")
	// ErrActivateFailed wraps an Applier.Activate failure; the live version is
	// left unchanged and the staged version remains pending.
	ErrActivateFailed = core.NewError("weightsync: activate failed")
	// ErrDrainTimeout is returned when in-flight work does not reach zero within
	// the drain timeout, so the swap is abandoned to avoid tearing a live
	// generation. The staged version remains pending.
	ErrDrainTimeout = core.NewError("weightsync: drain timed out with work in flight")
)

// drainPollInterval is how often the drained path re-checks the in-flight
// counter whilst waiting for a generation to quiesce.
const drainPollInterval = 2 * time.Millisecond

// Applier is the injected backend that performs the real weight apply. Stage
// loads the new weights for version into a staging buffer (identified by ref —
// e.g. a checkpoint path or content hash); Activate atomically swaps the staged
// weights into the live model.
//
//	type gpuApplier struct{ /* ... */ }
//	func (g *gpuApplier) Stage(ctx context.Context, v uint64, ref string) error { ... }
//	func (g *gpuApplier) Activate(ctx context.Context, v uint64) error { ... }
type Applier interface {
	// Stage loads the weights identified by ref into a staging buffer under the
	// given version. It must not affect the live model.
	Stage(ctx context.Context, version uint64, ref string) error
	// Activate atomically promotes the staged version to live.
	Activate(ctx context.Context, version uint64) error
}

// Coordinator drives an Applier through stage → activate, tracking the live
// version and any staged-but-not-yet-active version, and optionally draining
// in-flight work before a swap. It is safe for concurrent use.
//
//	co := weightsync.New(applier)
type Coordinator struct {
	applier Applier

	mu       sync.Mutex
	current  uint64 // live version (0 = none applied yet)
	pending  uint64 // staged-but-not-active version (0 = none)
	inFlight int64  // open generations gating a drained swap
}

// New returns a Coordinator over applier. The live version starts at zero, so
// the first accepted Update must use a version of at least one.
//
//	co := weightsync.New(applier)
func New(applier Applier) *Coordinator {
	return &Coordinator{applier: applier}
}

// Current returns the live weight version (0 before the first successful
// Update).
//
//	if co.Current() < want { co.Update(ctx, want, ref) }
func (c *Coordinator) Current() uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.current
}

// Pending returns the staged-but-not-active version, or 0 when nothing is
// staged. A non-zero Pending after an Update means staging succeeded but the
// swap did not — e.g. an activate error or a drain timeout.
//
//	if co.Pending() != 0 { co.retryActivate() }
func (c *Coordinator) Pending() uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.pending
}

// Begin marks the start of an in-flight generation (e.g. a request that reads
// the live weights). Pair every Begin with an End. UpdateDrained waits for the
// in-flight count to reach zero before swapping.
//
//	co.Begin()
//	defer co.End()
func (c *Coordinator) Begin() {
	c.mu.Lock()
	c.inFlight++
	c.mu.Unlock()
}

// End marks the completion of an in-flight generation. It never drives the
// counter below zero, so an over-call is harmless.
//
//	co.End()
func (c *Coordinator) End() {
	c.mu.Lock()
	if c.inFlight > 0 {
		c.inFlight--
	}
	c.mu.Unlock()
}

// InFlight returns the number of open generations.
//
//	if co.InFlight() == 0 { co.swapNow() }
func (c *Coordinator) InFlight() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.inFlight
}

// Update stages then immediately activates version, with no drain — the swap
// happens as soon as staging succeeds. Use this when there is no live
// generation to protect, or when an immediate swap is acceptable. For a swap
// that waits for in-flight work to quiesce, use UpdateDrained.
//
// A version less than or equal to Current is rejected with ErrStaleVersion. A
// Stage failure surfaces ErrStageFailed and leaves Current and Pending
// unchanged. An Activate failure surfaces ErrActivateFailed, leaves Current
// unchanged, and leaves the staged version in Pending for a later retry.
//
//	if err := co.Update(ctx, 2, "ckpt-iter-2"); err != nil { return err }
func (c *Coordinator) Update(ctx context.Context, version uint64, ref string) error {
	if err := c.stage(ctx, version, ref); err != nil {
		return err
	}
	return c.activate(ctx, version)
}

// UpdateDrained stages version, waits for in-flight work to reach zero (up to
// timeout), then activates — so a swap never tears a live generation. Staging
// happens up front; the wait sits between stage and activate.
//
// A version less than or equal to Current is rejected with ErrStaleVersion
// before any staging. If the drain does not complete within timeout the swap is
// abandoned with ErrDrainTimeout (the staged version stays in Pending). A
// cancelled ctx aborts the wait with the context error. A non-positive timeout
// means "wait indefinitely" (bounded only by ctx).
//
//	if err := co.UpdateDrained(ctx, 2, "ckpt-iter-2", 5*time.Second); err != nil { return err }
func (c *Coordinator) UpdateDrained(ctx context.Context, version uint64, ref string, timeout time.Duration) error {
	if err := c.stage(ctx, version, ref); err != nil {
		return err
	}
	if err := c.drain(ctx, timeout); err != nil {
		return err
	}
	return c.activate(ctx, version)
}

// stage validates the version against the live version and asks the Applier to
// load it. On success the version is recorded as pending.
func (c *Coordinator) stage(ctx context.Context, version uint64, ref string) error {
	c.mu.Lock()
	if version <= c.current {
		cur := c.current
		c.mu.Unlock()
		return core.E("weightsync.Update",
			core.Sprintf("version %d not newer than current %d", version, cur),
			ErrStaleVersion)
	}
	c.mu.Unlock()

	if err := c.applier.Stage(ctx, version, ref); err != nil {
		return core.E("weightsync.Update",
			core.Sprintf("stage version %d", version),
			core.ErrorJoin(ErrStageFailed, err))
	}

	c.mu.Lock()
	c.pending = version
	c.mu.Unlock()
	return nil
}

// activate asks the Applier to swap the staged version live. On success the live
// version advances and pending clears; on failure the version stays pending.
func (c *Coordinator) activate(ctx context.Context, version uint64) error {
	if err := c.applier.Activate(ctx, version); err != nil {
		return core.E("weightsync.Update",
			core.Sprintf("activate version %d", version),
			core.ErrorJoin(ErrActivateFailed, err))
	}

	c.mu.Lock()
	c.current = version
	if c.pending == version {
		c.pending = 0
	}
	c.mu.Unlock()
	return nil
}

// drain blocks until the in-flight count reaches zero, ctx is cancelled, or the
// timeout elapses. A non-positive timeout waits indefinitely (bounded by ctx).
func (c *Coordinator) drain(ctx context.Context, timeout time.Duration) error {
	if c.InFlight() == 0 {
		return nil
	}

	var deadline <-chan time.Time
	if timeout > 0 {
		t := time.NewTimer(timeout)
		defer t.Stop()
		deadline = t.C
	}

	ticker := time.NewTicker(drainPollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return core.E("weightsync.UpdateDrained", "drain cancelled", ctx.Err())
		case <-deadline:
			return core.E("weightsync.UpdateDrained",
				core.Sprintf("in-flight %d after drain", c.InFlight()),
				ErrDrainTimeout)
		case <-ticker.C:
			if c.InFlight() == 0 {
				return nil
			}
		}
	}
}
