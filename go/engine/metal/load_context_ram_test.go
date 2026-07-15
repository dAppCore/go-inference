// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
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
