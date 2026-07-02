// SPDX-Licence-Identifier: EUPL-1.2

package kvtier

import (
	"context"
	"errors"
	"testing"

	core "dappco.re/go"
)

// mb returns n mebibytes in bytes — keeps the budget tests readable against the
// per-tier KV-cache figures (the 16 GB GPU from RFC §6.2 holds only so
// many blocks before they spill to CPU then disk).
func mb(n int64) int64 { return n * 1024 * 1024 }

// move records one Store.Move call so a test can assert the exact offload/reload
// the policy asked for.
type move struct {
	id   string
	from Tier
	to   Tier
}

// fakeStore is the injected block mover. It records every Move in order and can
// be told to fail on the next call (failOn) to exercise the error path — the
// real Store copies bytes between GPU/CPU/disk; the policy only decides what to
// copy, so the test fake just remembers the plan.
//
//	fs := &fakeStore{}
//	m := New(Budget{GPU: mb(16), CPU: mb(64)}, fs)
//	_ = m.Put(context.Background(), Block{ID: "k0", SizeBytes: mb(8)})
//	// fs.moves now holds the demotions the placement required.
type fakeStore struct {
	moves []move
	// failOn fails the Move whose 1-based call index matches (0 = never).
	failOn int
	// failHop fails any Move matching this exact from→to hop (zero value = off),
	// letting a test target "the CPU→Disk cascade" regardless of call count.
	failHop *move
	calls   int
	failErr error
}

func (f *fakeStore) Move(_ context.Context, blockID string, from, to Tier) error {
	f.calls++
	hit := f.failOn != 0 && f.calls == f.failOn
	if f.failHop != nil && from == f.failHop.from && to == f.failHop.to {
		hit = true
	}
	if hit {
		if f.failErr != nil {
			return f.failErr
		}
		return core.E("test", "store move failed", nil)
	}
	f.moves = append(f.moves, move{id: blockID, from: from, to: to})
	return nil
}

// ---- Put ----------------------------------------------------------------

// TestKVTier_Put_Good covers the happy path: a fresh block lands on the GPU, a
// second block co-resides while both fit the GPU budget, and adding a third over
// budget demotes the least-recently-used block GPU→CPU (one recorded Move).
func TestKVTier_Put_Good(t *testing.T) {
	fs := &fakeStore{}
	m := New(Budget{GPU: mb(16), CPU: mb(64), Disk: mb(1024)}, fs)
	ctx := context.Background()

	if err := m.Put(ctx, Block{ID: "k0", SizeBytes: mb(8)}); err != nil {
		t.Fatalf("put k0: %v", err)
	}
	if got := m.TierOf("k0"); got != TierGPU {
		t.Fatalf("k0 tier: want GPU, got %v", got)
	}
	if len(fs.moves) != 0 {
		t.Fatalf("first put: want no moves, got %v", fs.moves)
	}

	// Second block: 8+8 = 16 ≤ 16 GPU budget, both stay on the GPU.
	if err := m.Put(ctx, Block{ID: "k1", SizeBytes: mb(8)}); err != nil {
		t.Fatalf("put k1: %v", err)
	}
	if got := m.TierOf("k1"); got != TierGPU {
		t.Fatalf("k1 tier: want GPU, got %v", got)
	}
	if len(fs.moves) != 0 {
		t.Fatalf("second put: want no moves, got %v", fs.moves)
	}

	// Third block over budget: 8+8+8 = 24 > 16 → demote LRU (k0) GPU→CPU.
	if err := m.Put(ctx, Block{ID: "k2", SizeBytes: mb(8)}); err != nil {
		t.Fatalf("put k2: %v", err)
	}
	if got := m.TierOf("k0"); got != TierCPU {
		t.Fatalf("k0 after demotion: want CPU, got %v", got)
	}
	if got := m.TierOf("k2"); got != TierGPU {
		t.Fatalf("k2 tier: want GPU, got %v", got)
	}
	if len(fs.moves) != 1 || fs.moves[0] != (move{id: "k0", from: TierGPU, to: TierCPU}) {
		t.Fatalf("want one demote k0 GPU->CPU, got %v", fs.moves)
	}
	// GPU now holds the two newest; CPU holds the spilled block.
	if got := m.Resident(TierGPU); len(got) != 2 {
		t.Fatalf("GPU resident: want 2, got %v", got)
	}
	if got := m.Resident(TierCPU); len(got) != 1 || got[0] != "k0" {
		t.Fatalf("CPU resident: want [k0], got %v", got)
	}
}

// TestKVTier_Put_Bad covers re-Put of an existing id (an in-place size update
// that re-demotes to honour the budget) and a zero/negative size being clamped
// rather than corrupting the accounting.
func TestKVTier_Put_Bad(t *testing.T) {
	fs := &fakeStore{}
	m := New(Budget{GPU: mb(16), CPU: mb(64), Disk: mb(1024)}, fs)
	ctx := context.Background()

	_ = m.Put(ctx, Block{ID: "a", SizeBytes: mb(4)})
	_ = m.Put(ctx, Block{ID: "b", SizeBytes: mb(4)})
	if len(fs.moves) != 0 {
		t.Fatalf("setup: want no moves, got %v", fs.moves)
	}

	// Re-Put a with a bigger size: 12+4 = 16 ≤ 16 still fits, no demotion, and
	// the re-Put refreshes recency so a is now MRU.
	if err := m.Put(ctx, Block{ID: "a", SizeBytes: mb(12)}); err != nil {
		t.Fatalf("re-put a: %v", err)
	}
	if got := m.TierOf("a"); got != TierGPU {
		t.Fatalf("a after re-put: want GPU, got %v", got)
	}
	if len(fs.moves) != 0 {
		t.Fatalf("re-put within budget: want no moves, got %v", fs.moves)
	}
	if n := len(m.Resident(TierGPU)); n != 2 {
		t.Fatalf("want 2 on GPU after re-put, got %d", n)
	}

	// Negative size is clamped to 0 — placement still succeeds, no spill.
	if err := m.Put(ctx, Block{ID: "c", SizeBytes: -5}); err != nil {
		t.Fatalf("put negative-size: %v", err)
	}
	if got := m.TierOf("c"); got != TierGPU {
		t.Fatalf("c tier: want GPU, got %v", got)
	}
}

// TestKVTier_Put_Ugly covers the oversized block: one larger than the GPU budget
// even on an empty GPU can never be placed and returns a typed ErrTooLarge with
// nothing moved, plus a duplicate-detectable wrapped message carrying the id.
func TestKVTier_Put_Ugly(t *testing.T) {
	fs := &fakeStore{}
	m := New(Budget{GPU: mb(16), CPU: mb(64), Disk: mb(1024)}, fs)
	ctx := context.Background()

	err := m.Put(ctx, Block{ID: "huge", SizeBytes: mb(32)})
	if err == nil {
		t.Fatalf("oversized block: want error, got nil")
	}
	if !errors.Is(err, ErrTooLarge) {
		t.Fatalf("oversized block: want ErrTooLarge, got %v", err)
	}
	if m.TierOf("huge") != TierNone {
		t.Fatalf("oversized block must not be resident, got %v", m.TierOf("huge"))
	}
	if len(fs.moves) != 0 {
		t.Fatalf("oversized block: want no moves, got %v", fs.moves)
	}
	if n := len(m.Resident(TierGPU)); n != 0 {
		t.Fatalf("GPU must stay empty after rejected put, got %d", n)
	}
}

// ---- Access -------------------------------------------------------------

// TestKVTier_Access_Good covers promotion: a block demoted to CPU is promoted
// back to the GPU on access (recorded CPU→GPU move), becomes most-recently-used,
// and a GPU-resident block accessed again is a no-op hit (no move).
func TestKVTier_Access_Good(t *testing.T) {
	fs := &fakeStore{}
	m := New(Budget{GPU: mb(16), CPU: mb(64), Disk: mb(1024)}, fs)
	ctx := context.Background()

	_ = m.Put(ctx, Block{ID: "k0", SizeBytes: mb(8)})
	_ = m.Put(ctx, Block{ID: "k1", SizeBytes: mb(8)})
	_ = m.Put(ctx, Block{ID: "k2", SizeBytes: mb(8)}) // demotes k0 -> CPU
	if m.TierOf("k0") != TierCPU {
		t.Fatalf("setup: k0 should be on CPU, got %v", m.TierOf("k0"))
	}
	fs.moves = nil // ignore setup moves; assert only the access plan

	// Access k0: promote CPU→GPU. GPU is full (k1,k2) so the LRU of those (k1)
	// is demoted GPU→CPU to make room.
	if err := m.Access(ctx, "k0"); err != nil {
		t.Fatalf("access k0: %v", err)
	}
	if got := m.TierOf("k0"); got != TierGPU {
		t.Fatalf("k0 after access: want GPU, got %v", got)
	}
	if got := m.TierOf("k1"); got != TierCPU {
		t.Fatalf("k1 should have been demoted to CPU, got %v", got)
	}
	wantMoves := map[move]bool{
		{id: "k1", from: TierGPU, to: TierCPU}: true,
		{id: "k0", from: TierCPU, to: TierGPU}: true,
	}
	if len(fs.moves) != 2 {
		t.Fatalf("access: want 2 moves, got %v", fs.moves)
	}
	for _, mv := range fs.moves {
		if !wantMoves[mv] {
			t.Fatalf("unexpected move %v (want %v)", mv, wantMoves)
		}
	}

	// Access a GPU-resident block: pure hit, no move, just recency bump.
	fs.moves = nil
	if err := m.Access(ctx, "k0"); err != nil {
		t.Fatalf("access resident k0: %v", err)
	}
	if len(fs.moves) != 0 {
		t.Fatalf("access GPU-resident: want no moves, got %v", fs.moves)
	}
}

// TestKVTier_Access_Bad covers pinning: a pinned GPU block is never demoted to
// make room for a promotion — an unpinned victim is chosen instead, and once
// every unpinned GPU block is gone the pinned ones stay put.
func TestKVTier_Access_Bad(t *testing.T) {
	fs := &fakeStore{}
	m := New(Budget{GPU: mb(16), CPU: mb(64), Disk: mb(1024)}, fs)
	ctx := context.Background()

	_ = m.Put(ctx, Block{ID: "pin", SizeBytes: mb(8)})
	_ = m.Put(ctx, Block{ID: "b", SizeBytes: mb(8)}) // GPU: pin, b
	m.Pin("pin")

	// A third block would normally demote the LRU (pin) — but it's pinned, so b
	// is demoted instead.
	_ = m.Put(ctx, Block{ID: "c", SizeBytes: mb(8)})
	if m.TierOf("pin") != TierGPU {
		t.Fatalf("pinned block must stay on GPU, got %v", m.TierOf("pin"))
	}
	if m.TierOf("b") != TierCPU {
		t.Fatalf("b should be demoted to CPU, got %v", m.TierOf("b"))
	}

	// Access b: promote it back. GPU holds pin (pinned) + c; only c is an
	// eligible victim, so c is demoted and pin is spared.
	fs.moves = nil
	if err := m.Access(ctx, "b"); err != nil {
		t.Fatalf("access b: %v", err)
	}
	if m.TierOf("pin") != TierGPU {
		t.Fatalf("pinned block must survive the promotion, got %v", m.TierOf("pin"))
	}
	if m.TierOf("b") != TierGPU {
		t.Fatalf("b should be promoted to GPU, got %v", m.TierOf("b"))
	}
	if m.TierOf("c") != TierCPU {
		t.Fatalf("c should be the demoted victim, got %v", m.TierOf("c"))
	}

	// Unpin then confirm it becomes an eviction candidate again.
	m.Unpin("pin")
	if m.IsPinned("pin") {
		t.Fatalf("pin should be unpinned now")
	}
}

// TestKVTier_Access_Ugly covers the unknown-id path: accessing a block the
// manager has never seen returns a typed ErrUnknownBlock and moves nothing.
func TestKVTier_Access_Ugly(t *testing.T) {
	fs := &fakeStore{}
	m := New(Budget{GPU: mb(16), CPU: mb(64), Disk: mb(1024)}, fs)
	ctx := context.Background()

	err := m.Access(ctx, "ghost")
	if err == nil {
		t.Fatalf("unknown id: want error, got nil")
	}
	if !errors.Is(err, ErrUnknownBlock) {
		t.Fatalf("unknown id: want ErrUnknownBlock, got %v", err)
	}
	if len(fs.moves) != 0 {
		t.Fatalf("unknown id: want no moves, got %v", fs.moves)
	}

	// Pin/Unpin/Remove/Evict on an unknown id are quiet no-ops (caller-friendly).
	m.Pin("ghost")
	m.Unpin("ghost")
	if err := m.Remove(ctx, "ghost"); err != nil {
		t.Fatalf("remove unknown: want nil, got %v", err)
	}
	if err := m.Evict(ctx, "ghost"); err != nil {
		t.Fatalf("evict unknown: want nil, got %v", err)
	}
}

// ---- Cascade ------------------------------------------------------------

// TestKVTier_Cascade_Good covers the GPU→CPU→Disk cascade: filling the GPU spills
// to CPU, then filling the CPU spills its LRU on to Disk, with each hop recorded
// as its own Move.
func TestKVTier_Cascade_Good(t *testing.T) {
	fs := &fakeStore{}
	// GPU holds 2 blocks, CPU holds 2 blocks; Disk is the backstop.
	m := New(Budget{GPU: mb(16), CPU: mb(16), Disk: mb(1024)}, fs)
	ctx := context.Background()

	// Put five 8 MB blocks. GPU keeps the two newest; the rest cascade down.
	for _, id := range []string{"k0", "k1", "k2", "k3", "k4"} {
		if err := m.Put(ctx, Block{ID: id, SizeBytes: mb(8)}); err != nil {
			t.Fatalf("put %s: %v", id, err)
		}
	}

	// GPU: the two most-recently-put (k3, k4).
	if got := m.Resident(TierGPU); len(got) != 2 {
		t.Fatalf("GPU: want 2 resident, got %v", got)
	}
	if m.TierOf("k4") != TierGPU || m.TierOf("k3") != TierGPU {
		t.Fatalf("newest two should be on GPU, got k3=%v k4=%v", m.TierOf("k3"), m.TierOf("k4"))
	}
	// CPU holds 2 (16 MB budget / 8 MB each); the oldest spilled to Disk.
	if got := m.Resident(TierCPU); len(got) != 2 {
		t.Fatalf("CPU: want 2 resident, got %v", got)
	}
	if m.TierOf("k0") != TierDisk {
		t.Fatalf("oldest block k0 should have cascaded to Disk, got %v", m.TierOf("k0"))
	}

	// The cascade recorded a k0 hop CPU→Disk somewhere in the move log.
	sawCascade := false
	for _, mv := range fs.moves {
		if mv.id == "k0" && mv.from == TierCPU && mv.to == TierDisk {
			sawCascade = true
		}
	}
	if !sawCascade {
		t.Fatalf("want a k0 CPU->Disk cascade move, got %v", fs.moves)
	}
}

// TestKVTier_Remove_Bad covers Evict/Remove of a block in a middle tier and
// the resulting freed budget: removing a CPU block frees CPU space so a later
// demotion no longer cascades to Disk.
func TestKVTier_Remove_Bad(t *testing.T) {
	fs := &fakeStore{}
	m := New(Budget{GPU: mb(16), CPU: mb(16), Disk: mb(1024)}, fs)
	ctx := context.Background()

	for _, id := range []string{"k0", "k1", "k2", "k3"} {
		_ = m.Put(ctx, Block{ID: id, SizeBytes: mb(8)})
	}
	// GPU: k2,k3  CPU: k0,k1 (full).
	if m.TierOf("k0") != TierCPU || m.TierOf("k1") != TierCPU {
		t.Fatalf("setup: k0,k1 should be on CPU, got k0=%v k1=%v", m.TierOf("k0"), m.TierOf("k1"))
	}

	// Remove k0 from CPU — frees a CPU slot, records a drop move CPU→TierNone.
	fs.moves = nil
	if err := m.Remove(ctx, "k0"); err != nil {
		t.Fatalf("remove k0: %v", err)
	}
	if m.TierOf("k0") != TierNone {
		t.Fatalf("k0 should be gone, got %v", m.TierOf("k0"))
	}
	if n := len(m.Resident(TierCPU)); n != 1 {
		t.Fatalf("CPU should hold 1 after remove, got %d", n)
	}

	// Now a new block demotes a GPU block to CPU — CPU has room (only k1), so
	// nothing cascades to Disk.
	if err := m.Put(ctx, Block{ID: "k4", SizeBytes: mb(8)}); err != nil {
		t.Fatalf("put k4: %v", err)
	}
	if n := len(m.Resident(TierDisk)); n != 0 {
		t.Fatalf("nothing should be on Disk yet, got %v", m.Resident(TierDisk))
	}

	// Evict (alias for drop) the GPU LRU explicitly.
	gpuBefore := len(m.Resident(TierGPU))
	victim := m.Resident(TierGPU)[0]
	if err := m.Evict(ctx, victim); err != nil {
		t.Fatalf("evict %s: %v", victim, err)
	}
	if len(m.Resident(TierGPU)) != gpuBefore-1 {
		t.Fatalf("evict should drop one GPU block")
	}
}

// TestKVTier_ErrStore_Ugly covers the Store failure path: when the injected store
// fails mid-cascade the operation surfaces the error and the manager's
// accounting is left unchanged (no partial placement).
func TestKVTier_ErrStore_Ugly(t *testing.T) {
	fs := &fakeStore{failOn: 1, failErr: core.E("test", "disk full", nil)}
	m := New(Budget{GPU: mb(8), CPU: mb(8), Disk: mb(1024)}, fs)
	ctx := context.Background()

	// First block lands on GPU with no move (Move call count still 0).
	if err := m.Put(ctx, Block{ID: "k0", SizeBytes: mb(8)}); err != nil {
		t.Fatalf("put k0: %v", err)
	}

	// Second block needs to demote k0 GPU→CPU — that is Move call #1, which the
	// fake fails. The Put must return the wrapped error and roll back so k1 is
	// NOT resident and k0 stays on the GPU.
	err := m.Put(ctx, Block{ID: "k1", SizeBytes: mb(8)})
	if err == nil {
		t.Fatalf("store failure: want error, got nil")
	}
	if !errors.Is(err, ErrStore) {
		t.Fatalf("store failure: want ErrStore, got %v", err)
	}
	if m.TierOf("k1") != TierNone {
		t.Fatalf("k1 must not be resident after a failed placement, got %v", m.TierOf("k1"))
	}
	if m.TierOf("k0") != TierGPU {
		t.Fatalf("k0 must stay on GPU after rollback, got %v", m.TierOf("k0"))
	}
}

// TestKVTier_Cascade_Rollback covers a mid-plan Store failure on a LATER hop:
// the GPU→CPU demotion succeeds, the cascading CPU→Disk hop fails, and the
// manager rolls the applied GPU→CPU hop back so the whole Put is undone and the
// pre-Put tier map is restored.
func TestKVTier_Cascade_Rollback(t *testing.T) {
	fs := &fakeStore{}
	// One block per bounded tier so any second/third block forces a cascade.
	m := New(Budget{GPU: mb(8), CPU: mb(8), Disk: mb(1024)}, fs)
	ctx := context.Background()

	if err := m.Put(ctx, Block{ID: "k0", SizeBytes: mb(8)}); err != nil {
		t.Fatalf("put k0: %v", err)
	}
	if err := m.Put(ctx, Block{ID: "k1", SizeBytes: mb(8)}); err != nil { // k0 -> CPU
		t.Fatalf("put k1: %v", err)
	}
	if m.TierOf("k0") != TierCPU || m.TierOf("k1") != TierGPU {
		t.Fatalf("setup: want k0=CPU k1=GPU, got k0=%v k1=%v", m.TierOf("k0"), m.TierOf("k1"))
	}

	// Arm the fake to fail any CPU→Disk hop. Putting k2 plans two hops:
	// k1 GPU→CPU (applied) then k0 CPU→Disk (fails) → rollback k1 back to GPU.
	fs.failHop = &move{from: TierCPU, to: TierDisk}
	fs.moves = nil
	err := m.Put(ctx, Block{ID: "k2", SizeBytes: mb(8)})
	if err == nil {
		t.Fatalf("cascade failure: want error, got nil")
	}
	if !errors.Is(err, ErrStore) {
		t.Fatalf("cascade failure: want ErrStore, got %v", err)
	}
	// Whole Put rolled back: k2 not resident, k1 back on GPU, k0 still on CPU.
	if m.TierOf("k2") != TierNone {
		t.Fatalf("k2 must not be resident after rollback, got %v", m.TierOf("k2"))
	}
	if m.TierOf("k1") != TierGPU {
		t.Fatalf("k1 must be rolled back to GPU, got %v", m.TierOf("k1"))
	}
	if m.TierOf("k0") != TierCPU {
		t.Fatalf("k0 must remain on CPU, got %v", m.TierOf("k0"))
	}
	// The rollback issued a compensating CPU→GPU move for k1.
	sawRollback := false
	for _, mv := range fs.moves {
		if mv.id == "k1" && mv.from == TierCPU && mv.to == TierGPU {
			sawRollback = true
		}
	}
	if !sawRollback {
		t.Fatalf("want a k1 CPU->GPU rollback move, got %v", fs.moves)
	}
}

// TestKVTier_Access_Rollback covers Access when the demotion it triggers fails:
// promoting a CPU block to a full GPU must demote a GPU victim, and if that
// demotion's Store hop fails the promoted block is returned to its old tier.
func TestKVTier_Access_StoreFail(t *testing.T) {
	fs := &fakeStore{}
	m := New(Budget{GPU: mb(8), CPU: mb(64), Disk: mb(1024)}, fs)
	ctx := context.Background()

	_ = m.Put(ctx, Block{ID: "k0", SizeBytes: mb(8)}) // GPU
	_ = m.Put(ctx, Block{ID: "k1", SizeBytes: mb(8)}) // k0 -> CPU, k1 on GPU
	if m.TierOf("k0") != TierCPU {
		t.Fatalf("setup: k0 should be on CPU, got %v", m.TierOf("k0"))
	}

	// Access k0 → promote to GPU, which demotes k1 GPU→CPU. Fail that demotion.
	fs.failHop = &move{from: TierGPU, to: TierCPU}
	err := m.Access(ctx, "k0")
	if err == nil {
		t.Fatalf("access demotion failure: want error, got nil")
	}
	if !errors.Is(err, ErrStore) {
		t.Fatalf("access demotion failure: want ErrStore, got %v", err)
	}
	// k0 returned to CPU, k1 untouched on GPU.
	if m.TierOf("k0") != TierCPU {
		t.Fatalf("k0 must revert to CPU after failed promote, got %v", m.TierOf("k0"))
	}
	if m.TierOf("k1") != TierGPU {
		t.Fatalf("k1 must remain on GPU, got %v", m.TierOf("k1"))
	}
}

// TestKVTier_Access_PromoteFail covers the case where the rebalance succeeds
// (the GPU has room, no victim needed) but the final promotion hop
// (CPU → GPU) itself fails: the block reverts to its source tier and ErrStore
// is returned.
func TestKVTier_Access_PromoteFail(t *testing.T) {
	fs := &fakeStore{}
	m := New(Budget{GPU: mb(16), CPU: mb(64), Disk: mb(1024)}, fs)
	ctx := context.Background()

	_ = m.Put(ctx, Block{ID: "a", SizeBytes: mb(8)})
	_ = m.Put(ctx, Block{ID: "b", SizeBytes: mb(8)})
	_ = m.Put(ctx, Block{ID: "c", SizeBytes: mb(8)}) // a -> CPU
	if m.TierOf("a") != TierCPU {
		t.Fatalf("setup: a should be on CPU, got %v", m.TierOf("a"))
	}
	// Free a GPU slot so the promote of a needs no demotion (pure promote hop).
	_ = m.Remove(ctx, "b")
	if n := len(m.Resident(TierGPU)); n != 1 {
		t.Fatalf("setup: GPU should hold 1 (c), got %d", n)
	}

	fs.failHop = &move{from: TierCPU, to: TierGPU}
	err := m.Access(ctx, "a")
	if err == nil {
		t.Fatalf("promote hop failure: want error, got nil")
	}
	if !errors.Is(err, ErrStore) {
		t.Fatalf("promote hop failure: want ErrStore, got %v", err)
	}
	if m.TierOf("a") != TierCPU {
		t.Fatalf("a must revert to CPU after failed promote, got %v", m.TierOf("a"))
	}
}

// TestKVTier_Remove_StoreFail covers Remove when the Store fails to free the
// block: the error is surfaced as ErrStore and the block stays tracked.
func TestKVTier_Remove_StoreFail(t *testing.T) {
	fs := &fakeStore{}
	m := New(Budget{GPU: mb(16), CPU: mb(64), Disk: mb(1024)}, fs)
	ctx := context.Background()

	_ = m.Put(ctx, Block{ID: "a", SizeBytes: mb(4)})
	fs.failHop = &move{from: TierGPU, to: TierNone}
	err := m.Remove(ctx, "a")
	if err == nil {
		t.Fatalf("remove store failure: want error, got nil")
	}
	if !errors.Is(err, ErrStore) {
		t.Fatalf("remove store failure: want ErrStore, got %v", err)
	}
	if m.TierOf("a") != TierGPU {
		t.Fatalf("a must remain tracked after failed remove, got %v", m.TierOf("a"))
	}
}

// ---- small surface coverage --------------------------------------------

// TestKVTier_String_Good exercises the remaining accessors and the Tier.String
// helper so the public surface is fully covered.
func TestKVTier_String_Good(t *testing.T) {
	fs := &fakeStore{}
	m := New(Budget{GPU: mb(16), CPU: mb(64), Disk: mb(1024)}, fs)
	ctx := context.Background()

	// Tier.String for diagnostics.
	for tier, want := range map[Tier]string{
		TierGPU:  "gpu",
		TierCPU:  "cpu",
		TierDisk: "disk",
		TierNone: "none",
		Tier(99): "unknown",
	} {
		if got := tier.String(); got != want {
			t.Fatalf("Tier(%d).String() = %q, want %q", tier, got, want)
		}
	}

	_ = m.Put(ctx, Block{ID: "a", SizeBytes: mb(4)})
	if !m.IsResident("a") {
		t.Fatalf("a should be resident")
	}
	if m.IsResident("nope") {
		t.Fatalf("nope should not be resident")
	}
	if m.IsPinned("a") {
		t.Fatalf("a should not be pinned yet")
	}
	m.Pin("a")
	if !m.IsPinned("a") {
		t.Fatalf("a should be pinned")
	}

	// Resident on an empty/unknown tier returns an empty slice, not nil-panic.
	if got := m.Resident(Tier(99)); len(got) != 0 {
		t.Fatalf("unknown tier resident: want empty, got %v", got)
	}

	// Len reports the total tracked blocks across all tiers.
	if m.Len() != 1 {
		t.Fatalf("Len: want 1, got %d", m.Len())
	}
}

// TestKVTier_New_Ugly covers budget clamping: negative budgets are floored to 0,
// and a Put on a zero-GPU manager is rejected as too large (nothing fits).
func TestKVTier_New_Ugly(t *testing.T) {
	fs := &fakeStore{}
	m := New(Budget{GPU: -1, CPU: -1, Disk: -1}, fs)
	ctx := context.Background()

	err := m.Put(ctx, Block{ID: "x", SizeBytes: mb(1)})
	if !errors.Is(err, ErrTooLarge) {
		t.Fatalf("zero-GPU put: want ErrTooLarge, got %v", err)
	}

	// A zero-size block fits even a zero budget (0 ≤ 0) and lands on GPU.
	if err := m.Put(ctx, Block{ID: "empty", SizeBytes: 0}); err != nil {
		t.Fatalf("zero-size put: %v", err)
	}
	if m.TierOf("empty") != TierGPU {
		t.Fatalf("zero-size block should be on GPU, got %v", m.TierOf("empty"))
	}
}
