// SPDX-Licence-Identifier: EUPL-1.2

package weightsync

import (
	"context"
	"sync"
	"testing"
	"time"

	core "dappco.re/go"
)

// fakeApplier records the order of Stage/Activate calls and can be told to fail
// at a chosen step, modelling a GPU weight-apply backend in tests.
//
//	a := &fakeApplier{}
//	co := New(a)
//	co.Update(context.Background(), 1, "ckpt-iter-1")
type fakeApplier struct {
	mu sync.Mutex

	staged    []uint64 // versions handed to Stage, in order
	activated []uint64 // versions handed to Activate, in order
	refs      []string // refs handed to Stage, in order

	stageErr    error // non-nil → Stage returns it
	activateErr error // non-nil → Activate returns it

	// onActivate, if set, runs at the top of Activate before the error check —
	// used to drive in-flight work whilst a swap is mid-flight.
	onActivate func()
}

func (a *fakeApplier) Stage(_ context.Context, version uint64, ref string) error {
	a.mu.Lock()
	a.staged = append(a.staged, version)
	a.refs = append(a.refs, ref)
	a.mu.Unlock()
	return a.stageErr
}

func (a *fakeApplier) Activate(_ context.Context, version uint64) error {
	if a.onActivate != nil {
		a.onActivate()
	}
	if a.activateErr != nil {
		return a.activateErr
	}
	a.mu.Lock()
	a.activated = append(a.activated, version)
	a.mu.Unlock()
	return nil
}

func (a *fakeApplier) stagedSnapshot() []uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]uint64(nil), a.staged...)
}

func (a *fakeApplier) activatedSnapshot() []uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]uint64(nil), a.activated...)
}

// --- Update -----------------------------------------------------------------

// TestWeightSync_Update_Good proves stage→activate advances the live version and
// drives the applier in the right order.
func TestWeightSync_Update_Good(t *testing.T) {
	a := &fakeApplier{}
	co := New(a)

	if got := co.Current(); got != 0 {
		t.Fatalf("fresh coordinator Current() = %d, want 0", got)
	}

	if err := co.Update(context.Background(), 1, "ckpt-iter-1"); err != nil {
		t.Fatalf("Update(1) error: %v", err)
	}
	if got := co.Current(); got != 1 {
		t.Fatalf("Current() = %d, want 1", got)
	}
	if err := co.Update(context.Background(), 2, "ckpt-iter-2"); err != nil {
		t.Fatalf("Update(2) error: %v", err)
	}
	if got := co.Current(); got != 2 {
		t.Fatalf("Current() = %d, want 2", got)
	}

	if want := []uint64{1, 2}; !equalU64(a.stagedSnapshot(), want) {
		t.Fatalf("staged = %v, want %v", a.stagedSnapshot(), want)
	}
	if want := []uint64{1, 2}; !equalU64(a.activatedSnapshot(), want) {
		t.Fatalf("activated = %v, want %v", a.activatedSnapshot(), want)
	}
	if want := []string{"ckpt-iter-1", "ckpt-iter-2"}; !equalStr(a.refs, want) {
		t.Fatalf("refs = %v, want %v", a.refs, want)
	}
	// A successful Update leaves nothing pending.
	if got := co.Pending(); got != 0 {
		t.Fatalf("Pending() after success = %d, want 0", got)
	}
}

// TestWeightSync_Update_Bad proves a stale or duplicate version is rejected with
// a typed error and never touches the applier or the live version.
func TestWeightSync_Update_Bad(t *testing.T) {
	a := &fakeApplier{}
	co := New(a)

	if err := co.Update(context.Background(), 5, "ckpt-iter-5"); err != nil {
		t.Fatalf("Update(5) error: %v", err)
	}

	// Equal version → stale (duplicate).
	err := co.Update(context.Background(), 5, "again")
	if err == nil {
		t.Fatal("Update(5) again: want stale error, got nil")
	}
	if !core.Is(err, ErrStaleVersion) {
		t.Fatalf("Update(5) again: error %v, want ErrStaleVersion", err)
	}

	// Lower version → stale.
	err = co.Update(context.Background(), 3, "older")
	if err == nil {
		t.Fatal("Update(3): want stale error, got nil")
	}
	if !core.Is(err, ErrStaleVersion) {
		t.Fatalf("Update(3): error %v, want ErrStaleVersion", err)
	}

	// Zero version is never valid (Current starts at 0, so 0 <= 0).
	err = co.Update(context.Background(), 0, "zero")
	if !core.Is(err, ErrStaleVersion) {
		t.Fatalf("Update(0): error %v, want ErrStaleVersion", err)
	}

	if got := co.Current(); got != 5 {
		t.Fatalf("Current() after stale rejects = %d, want 5", got)
	}
	// Only the first, accepted update reached the applier.
	if want := []uint64{5}; !equalU64(a.stagedSnapshot(), want) {
		t.Fatalf("staged = %v, want %v", a.stagedSnapshot(), want)
	}
	if want := []uint64{5}; !equalU64(a.activatedSnapshot(), want) {
		t.Fatalf("activated = %v, want %v", a.activatedSnapshot(), want)
	}
}

// TestWeightSync_Update_Ugly proves a Stage failure leaves current unchanged and
// nothing pending, and an Activate failure surfaces a typed error whilst leaving
// the live version unchanged.
func TestWeightSync_Update_Ugly(t *testing.T) {
	// Stage failure path.
	stageBoom := core.NewError("staging buffer full")
	a := &fakeApplier{stageErr: stageBoom}
	co := New(a)

	err := co.Update(context.Background(), 1, "ckpt-iter-1")
	if err == nil {
		t.Fatal("Update with stage error: want error, got nil")
	}
	if !core.Is(err, ErrStageFailed) {
		t.Fatalf("stage error %v, want ErrStageFailed", err)
	}
	if !core.Is(err, stageBoom) {
		t.Fatalf("stage error %v, want wrapped cause %v", err, stageBoom)
	}
	if got := co.Current(); got != 0 {
		t.Fatalf("Current() after stage fail = %d, want 0", got)
	}
	if got := co.Pending(); got != 0 {
		t.Fatalf("Pending() after stage fail = %d, want 0", got)
	}
	if len(a.activatedSnapshot()) != 0 {
		t.Fatalf("activated should be empty after stage fail, got %v", a.activatedSnapshot())
	}

	// Activate failure path.
	activateBoom := core.NewError("device swap rejected")
	b := &fakeApplier{activateErr: activateBoom}
	cb := New(b)

	// Seed a good live version first so we can prove it does not change.
	if err := cb.Update(context.Background(), 1, "ok"); err == nil {
		t.Fatal("seed Update should fail because Activate always errors")
	}
	// Live version stayed at 0; the staged version is surfaced as pending.
	if got := cb.Current(); got != 0 {
		t.Fatalf("Current() after activate fail = %d, want 0", got)
	}
	err = cb.Update(context.Background(), 2, "ckpt-iter-2")
	if err == nil {
		t.Fatal("Update with activate error: want error, got nil")
	}
	if !core.Is(err, ErrActivateFailed) {
		t.Fatalf("activate error %v, want ErrActivateFailed", err)
	}
	if !core.Is(err, activateBoom) {
		t.Fatalf("activate error %v, want wrapped cause %v", err, activateBoom)
	}
	if got := cb.Current(); got != 0 {
		t.Fatalf("Current() after activate fail = %d, want 0", got)
	}
	// Staging happened, so the version sits pending awaiting a future activate.
	if got := cb.Pending(); got != 2 {
		t.Fatalf("Pending() after activate fail = %d, want 2", got)
	}
}

// --- Drain ------------------------------------------------------------------

// TestWeightSync_Drain_Good proves UpdateDrained waits for in-flight work to
// reach zero before activating, so a swap never tears a live generation.
func TestWeightSync_Drain_Good(t *testing.T) {
	a := &fakeApplier{}
	co := New(a)

	// Open a live generation.
	co.Begin()
	co.Begin()
	if got := co.InFlight(); got != 2 {
		t.Fatalf("InFlight() = %d, want 2", got)
	}

	// Activate must observe in-flight == 0 at the moment of the swap.
	var observed int64 = -1
	a.onActivate = func() { observed = co.InFlight() }

	done := make(chan error, 1)
	go func() {
		done <- co.UpdateDrained(context.Background(), 1, "ckpt-iter-1", 2*time.Second)
	}()

	// Give the goroutine time to stage and start waiting on the drain.
	time.Sleep(20 * time.Millisecond)
	// Whilst work is in flight, no activate may have happened yet.
	if got := co.Current(); got != 0 {
		t.Fatalf("Current() before drain completes = %d, want 0", got)
	}

	// Drain the generation.
	co.End()
	co.End()

	if err := <-done; err != nil {
		t.Fatalf("UpdateDrained error: %v", err)
	}
	if got := co.Current(); got != 1 {
		t.Fatalf("Current() = %d, want 1", got)
	}
	if observed != 0 {
		t.Fatalf("in-flight at activate = %d, want 0", observed)
	}

	// A drain with zero in-flight activates immediately.
	if err := co.UpdateDrained(context.Background(), 2, "ckpt-iter-2", time.Second); err != nil {
		t.Fatalf("UpdateDrained(2) error: %v", err)
	}
	if got := co.Current(); got != 2 {
		t.Fatalf("Current() = %d, want 2", got)
	}
}

// TestWeightSync_Drain_Bad proves the drain times out (typed error) when
// in-flight work never reaches zero, and leaves the live version unchanged.
func TestWeightSync_Drain_Bad(t *testing.T) {
	a := &fakeApplier{}
	co := New(a)

	co.Begin() // never ended → drain can never complete

	err := co.UpdateDrained(context.Background(), 1, "ckpt-iter-1", 30*time.Millisecond)
	if err == nil {
		t.Fatal("UpdateDrained: want drain timeout, got nil")
	}
	if !core.Is(err, ErrDrainTimeout) {
		t.Fatalf("drain error %v, want ErrDrainTimeout", err)
	}
	if got := co.Current(); got != 0 {
		t.Fatalf("Current() after drain timeout = %d, want 0", got)
	}
	// Staged but never activated → pending holds the version.
	if got := co.Pending(); got != 1 {
		t.Fatalf("Pending() after drain timeout = %d, want 1", got)
	}
	// Activate must not have run.
	if len(a.activatedSnapshot()) != 0 {
		t.Fatalf("activated should be empty after drain timeout, got %v", a.activatedSnapshot())
	}
}

// TestWeightSync_Drain_Ugly proves the drain path also honours staleness and a
// cancelled context, and that End never drives the counter below zero.
func TestWeightSync_Drain_Ugly(t *testing.T) {
	a := &fakeApplier{}
	co := New(a)

	if err := co.UpdateDrained(context.Background(), 4, "ckpt-iter-4", time.Second); err != nil {
		t.Fatalf("UpdateDrained(4) error: %v", err)
	}

	// Stale version rejected before any staging on the drain path too.
	err := co.UpdateDrained(context.Background(), 4, "again", time.Second)
	if !core.Is(err, ErrStaleVersion) {
		t.Fatalf("UpdateDrained(4) again: error %v, want ErrStaleVersion", err)
	}

	// A cancelled context aborts the drain wait with the context error.
	co.Begin()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err = co.UpdateDrained(ctx, 5, "ckpt-iter-5", time.Second)
	if err == nil {
		t.Fatal("UpdateDrained with cancelled ctx: want error, got nil")
	}
	if !core.Is(err, context.Canceled) {
		t.Fatalf("cancelled drain error %v, want context.Canceled", err)
	}
	if got := co.Current(); got != 4 {
		t.Fatalf("Current() after cancelled drain = %d, want 4", got)
	}
	co.End()

	// End must clamp at zero — extra Ends never make InFlight negative.
	co.End()
	co.End()
	if got := co.InFlight(); got != 0 {
		t.Fatalf("InFlight() after over-End = %d, want 0", got)
	}
}

// --- Current / Pending ------------------------------------------------------

// TestWeightSync_Current_Good proves Current and Pending reflect a normal
// stage→activate lifecycle and are concurrency-safe under load.
func TestWeightSync_Current_Good(t *testing.T) {
	a := &fakeApplier{}
	co := New(a)

	if got := co.Current(); got != 0 {
		t.Fatalf("Current() = %d, want 0", got)
	}
	if got := co.Pending(); got != 0 {
		t.Fatalf("Pending() = %d, want 0", got)
	}

	if err := co.Update(context.Background(), 7, "ckpt-iter-7"); err != nil {
		t.Fatalf("Update(7) error: %v", err)
	}
	if got := co.Current(); got != 7 {
		t.Fatalf("Current() = %d, want 7", got)
	}
	if got := co.Pending(); got != 0 {
		t.Fatalf("Pending() = %d, want 0", got)
	}

	// Concurrent readers race-free against Begin/End churn.
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			co.Begin()
			_ = co.Current()
			_ = co.Pending()
			_ = co.InFlight()
			co.End()
		}()
	}
	wg.Wait()
	if got := co.InFlight(); got != 0 {
		t.Fatalf("InFlight() after churn = %d, want 0", got)
	}
	if got := co.Current(); got != 7 {
		t.Fatalf("Current() after churn = %d, want 7", got)
	}
}

// TestWeightSync_Current_Bad proves a rejected (stale) update does not disturb
// Current or Pending.
func TestWeightSync_Current_Bad(t *testing.T) {
	a := &fakeApplier{}
	co := New(a)

	if err := co.Update(context.Background(), 9, "ckpt-iter-9"); err != nil {
		t.Fatalf("Update(9) error: %v", err)
	}
	_ = co.Update(context.Background(), 1, "stale") // rejected

	if got := co.Current(); got != 9 {
		t.Fatalf("Current() = %d, want 9", got)
	}
	if got := co.Pending(); got != 0 {
		t.Fatalf("Pending() = %d, want 0", got)
	}
}

// TestWeightSync_Current_Ugly proves Pending tracks a staged-but-not-yet-active
// version after an Activate failure, then clears once a later Update succeeds.
func TestWeightSync_Current_Ugly(t *testing.T) {
	boom := core.NewError("swap rejected once")
	a := &fakeApplier{activateErr: boom}
	co := New(a)

	// Activate fails → version 3 is staged but not live.
	_ = co.Update(context.Background(), 3, "ckpt-iter-3")
	if got := co.Current(); got != 0 {
		t.Fatalf("Current() = %d, want 0", got)
	}
	if got := co.Pending(); got != 3 {
		t.Fatalf("Pending() = %d, want 3", got)
	}

	// Recover the applier; a fresh, higher update now succeeds and clears pending.
	a.mu.Lock()
	a.activateErr = nil
	a.mu.Unlock()

	if err := co.Update(context.Background(), 4, "ckpt-iter-4"); err != nil {
		t.Fatalf("Update(4) error: %v", err)
	}
	if got := co.Current(); got != 4 {
		t.Fatalf("Current() = %d, want 4", got)
	}
	if got := co.Pending(); got != 0 {
		t.Fatalf("Pending() = %d, want 0", got)
	}
}

// --- helpers ----------------------------------------------------------------

func equalU64(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalStr(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
