// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"testing"

	"dappco.re/go/inference/model"
)

// ramGuardArch builds a small mixed sliding/global arch: 4 global owners +
// 8 sliding owners, kvDim 512 (2 heads × 256), window 1024 — per-row global
// cost 4 × 512 × 2B × 2 = 8KiB.
func ramGuardArch() model.Arch {
	types := make([]string, 12)
	for i := range types {
		if i%3 == 0 {
			types[i] = "full_attention"
		} else {
			types[i] = "sliding_attention"
		}
	}
	return model.Arch{
		Layer: model.DeriveLayers(types, 0), KVHeads: 2, HeadDim: 256, SlidingWindow: 1024,
	}
}

func TestClampContextToRAMKeepsDefaultWhenBoxFits(t *testing.T) {
	arch := ramGuardArch()
	// 96GiB box, 3GiB weights: budget ≈ 69.8GiB, per-row 8KiB → millions of rows.
	if got := clampContextToRAM(131072, arch, 3<<30, 96<<30); got != 131072 {
		t.Fatalf("roomy box must keep the default: got %d", got)
	}
}

func TestClampContextToRAMClampsSmallBox(t *testing.T) {
	arch := ramGuardArch()
	// 16GiB box, 4GiB weights: reserve 8GiB + fixed 4GiB + weights 4GiB = 16GiB
	// leaves nothing — floor. Give it 24GiB: budget = 24-8-4-4 = 8GiB minus the
	// sliding rings (8 × 1024 rows × 2KiB = 16MiB) → ~8GiB / 8KiB ≈ 1M rows (no
	// clamp at 128K); tighten weights to force the clamp arithmetic instead.
	got := clampContextToRAM(131072, arch, 11<<30, 24<<30)
	// budget = 24-8-4-11 = 1GiB - rings(16MiB) → ≈ 131,006 rows... still ~128K.
	// Halve again: 0.5GiB budget → ~64K rows.
	got = clampContextToRAM(131072, arch, 11<<30+512<<20, 24<<30)
	if got >= 131072 || got < 4096 {
		t.Fatalf("tight box must clamp inside [4096, default): got %d", got)
	}
	if got%1024 != 0 {
		t.Fatalf("clamped context must land on a 1024 boundary: got %d", got)
	}
}

func TestClampContextToRAMFloorsWhenWeightsCrowdTheBox(t *testing.T) {
	arch := ramGuardArch()
	// 16GiB box, 17GiB mapped weights: reserve+weights+fixed exceeds RAM.
	if got := clampContextToRAM(131072, arch, 17<<30, 16<<30); got != 4096 {
		t.Fatalf("crowded box must keep the 4096 floor: got %d", got)
	}
}

func TestClampContextToRAMDisabledPaths(t *testing.T) {
	arch := ramGuardArch()
	if got := clampContextToRAM(131072, arch, 17<<30, 0); got != 131072 {
		t.Fatalf("failed RAM probe must not clamp: got %d", got)
	}
	if got := clampContextToRAM(4096, arch, 17<<30, 16<<30); got != 4096 {
		t.Fatalf("floor-sized default must pass through: got %d", got)
	}
	old := contextRAMGuardEnabled
	contextRAMGuardEnabled = false
	defer func() { contextRAMGuardEnabled = old }()
	if got := clampContextToRAM(131072, arch, 17<<30, 16<<30); got != 131072 {
		t.Fatalf("kill switch must disable the clamp: got %d", got)
	}
}

func TestClampContextToRAMSlidingChargedFixed(t *testing.T) {
	// All-global vs mostly-sliding at the same layer count: the sliding arch
	// must afford a larger context from the same budget (rings don't scale).
	globalTypes := make([]string, 12)
	for i := range globalTypes {
		globalTypes[i] = "full_attention"
	}
	allGlobal := model.Arch{Layer: model.DeriveLayers(globalTypes, 0), KVHeads: 2, HeadDim: 256}
	mixed := ramGuardArch()
	ram, weights := uint64(24<<30), uint64(11<<30)
	g := clampContextToRAM(1<<20, allGlobal, weights, ram)
	m := clampContextToRAM(1<<20, mixed, weights, ram)
	if m <= g {
		t.Fatalf("sliding-heavy arch must afford more context than all-global: mixed %d <= global %d", m, g)
	}
}

// TestKvRowBytes_Good pins the per-row formula (kvHeads × headDim × bf16Size ×
// 2 for K+V) against a layer that declares its OWN geometry, overriding the
// arch-level fallback.
func TestKvRowBytes_Good(t *testing.T) {
	arch := model.Arch{HeadDim: 128, KVHeads: 8}
	spec := model.LayerSpec{HeadDim: 256, KVHeads: 2}
	want := uint64(2*256) * bf16Size * 2
	if got := kvRowBytes(spec, arch); got != want {
		t.Fatalf("kvRowBytes(per-layer geometry) = %d, want %d", got, want)
	}
}

// TestKvRowBytes_Ugly is the surprising-but-valid case: a layer that declares
// NO geometry of its own falls back to the arch-level HeadDim/KVHeads
// (headDimOf/kvHeadsOf's fallback), rather than costing zero.
func TestKvRowBytes_Ugly(t *testing.T) {
	arch := model.Arch{HeadDim: 128, KVHeads: 8}
	spec := model.LayerSpec{}
	want := uint64(8*128) * bf16Size * 2
	if got := kvRowBytes(spec, arch); got != want {
		t.Fatalf("kvRowBytes(no per-layer geometry) = %d, want %d (arch fallback)", got, want)
	}
}

// TestSessionMemoryBudgetBytes_Good pins the exact subtraction — physical RAM
// minus reserve minus weights minus the fixed slab — clampContextToRAM itself
// solves against, on a box with room to spare.
func TestSessionMemoryBudgetBytes_Good(t *testing.T) {
	const ramBytes, weightsBytes = uint64(64 << 30), uint64(3 << 30)
	reserve := min(max(ramBytes/5, 8<<30), 24<<30)
	want := ramBytes - reserve - weightsBytes - (4 << 30)
	got, ok := sessionMemoryBudgetBytes(weightsBytes, ramBytes)
	if !ok {
		t.Fatalf("sessionMemoryBudgetBytes() ok = false, want true (roomy box)")
	}
	if got != want {
		t.Fatalf("sessionMemoryBudgetBytes() = %d, want %d", got, want)
	}
}

// TestSessionMemoryBudgetBytes_Bad is the fail-closed case: weights alone
// (plus the fixed reserve) already exceed physical RAM, so there is no room
// for any KV bytes at all.
func TestSessionMemoryBudgetBytes_Bad(t *testing.T) {
	got, ok := sessionMemoryBudgetBytes(17<<30, 16<<30)
	if ok {
		t.Fatalf("sessionMemoryBudgetBytes() ok = true, want false (weights crowd the box)")
	}
	if got != 0 {
		t.Fatalf("sessionMemoryBudgetBytes() budget = %d, want 0 when declined", got)
	}
}

// TestSessionMemoryBudgetBytes_Ugly pins the inclusive boundary: ramBytes
// exactly equal to reserve+weights+fixed still declines (<=, not <) — one
// byte short of room is treated the same as none.
func TestSessionMemoryBudgetBytes_Ugly(t *testing.T) {
	const weightsBytes = uint64(4 << 30)
	reserve := uint64(8 << 30) // min(max(ramBytes/5,8GiB),24GiB) at this ramBytes
	ramBytes := reserve + weightsBytes + (4 << 30)
	if _, ok := sessionMemoryBudgetBytes(weightsBytes, ramBytes); ok {
		t.Fatalf("sessionMemoryBudgetBytes() ok = true at the exact boundary, want false (inclusive <=)")
	}
}

// TestSessionKVBytesAt_Good sums an all-global arch's KV cost linearly across
// every cache-owning layer: ctxLen × Σ kvRowBytes.
func TestSessionKVBytesAt_Good(t *testing.T) {
	globalTypes := make([]string, 12)
	for i := range globalTypes {
		globalTypes[i] = "full_attention"
	}
	arch := model.Arch{Layer: model.DeriveLayers(globalTypes, 0), KVHeads: 2, HeadDim: 256}
	const ctxLen = 4096
	rowBytes := uint64(2*256) * bf16Size * 2
	want := uint64(len(globalTypes)) * uint64(ctxLen) * rowBytes
	if got := sessionKVBytesAt(arch, ctxLen); got != want {
		t.Fatalf("sessionKVBytesAt(all-global) = %d, want %d", got, want)
	}
}

// TestSessionKVBytesAt_Bad reports zero for a non-positive context length —
// there is no session to cost.
func TestSessionKVBytesAt_Bad(t *testing.T) {
	arch := ramGuardArch()
	if got := sessionKVBytesAt(arch, 0); got != 0 {
		t.Fatalf("sessionKVBytesAt(ctxLen=0) = %d, want 0", got)
	}
	if got := sessionKVBytesAt(arch, -1); got != 0 {
		t.Fatalf("sessionKVBytesAt(ctxLen=-1) = %d, want 0", got)
	}
}

// TestSessionKVBytesAt_Ugly is the forward-direction twin of
// TestClampContextToRAMSlidingChargedFixed: at a context length past the
// sliding window, a sliding layer's contribution caps at SlidingWindow rows
// while a global layer's keeps scaling with ctxLen — cross-checked against
// ramGuardArch's own documented per-row/ring numbers (8KiB global row total,
// 16MiB total sliding-ring charge) so the forward accounting matches the
// clampContextToRAM comments exactly.
func TestSessionKVBytesAt_Ugly(t *testing.T) {
	arch := ramGuardArch() // 4 global owners + 8 sliding owners, window 1024, row 2KiB
	const ctxLen = 131072  // well past the 1024 window
	wantGlobal := uint64(4) * uint64(ctxLen) * (2 << 10)     // 4 owners × ctxLen rows × 2KiB/row
	wantSliding := uint64(8) * uint64(1024) * (2 << 10)      // 8 owners × window(1024) rows × 2KiB/row
	want := wantGlobal + wantSliding
	if got := sessionKVBytesAt(arch, ctxLen); got != want {
		t.Fatalf("sessionKVBytesAt(mixed, ctxLen=%d) = %d, want %d (global %d + sliding-ring %d)",
			ctxLen, got, want, wantGlobal, wantSliding)
	}
}

// TestIntFromBytes_Good passes a realistic byte count through unchanged.
func TestIntFromBytes_Good(t *testing.T) {
	if got := intFromBytes(4096); got != 4096 {
		t.Fatalf("intFromBytes(4096) = %d, want 4096", got)
	}
}

// TestIntFromBytes_Ugly saturates a value beyond the platform int range
// instead of wrapping negative — the theoretical-overflow guard.
func TestIntFromBytes_Ugly(t *testing.T) {
	if got := intFromBytes(math.MaxUint64); got != math.MaxInt {
		t.Fatalf("intFromBytes(MaxUint64) = %d, want math.MaxInt (saturated)", got)
	}
}
