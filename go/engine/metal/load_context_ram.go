// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"golang.org/x/sys/unix"
)

// contextRAMGuardEnabled gates the RAM-aware clamp on the DEFAULT context
// (LTHN_CONTEXT_RAM_GUARD=0 disables — the kill switch every adaptive
// behaviour carries). An explicit -context is never touched by this guard.
var contextRAMGuardEnabled = os.Getenv("LTHN_CONTEXT_RAM_GUARD") != "0"

// physicalRAMBytes reports the box's physical memory (sysctl hw.memsize);
// 0 when the probe fails, which disables the clamp rather than guessing.
func physicalRAMBytes() uint64 {
	v, err := unix.SysctlUint64("hw.memsize")
	if err != nil {
		return 0
	}
	return v
}

// kvRowBytes is one cache-owning layer's per-row KV cost: K+V at the bf16
// rate (2 × kvDim × 2B) — deliberately ignoring the q8 saving, because the q8
// GEMM-prefix mirrors transiently return those layers to bf16 size during a
// long prefill. Shared by clampContextToRAM (which solves budget → rows) and
// sessionKVBytesAt (which solves rows → budget) so the two directions read
// from ONE formula and can never drift apart.
func kvRowBytes(spec model.LayerSpec, arch model.Arch) uint64 {
	lhd, lkv := headDimOf(spec, arch.HeadDim), kvHeadsOf(spec, arch.KVHeads)
	return uint64(lkv*lhd) * bf16Size * 2
}

// sessionMemoryBudgetBytes is the KV-row budget clampContextToRAM (and the
// batch-lane admission gate, lane_set.go) solve against: physical RAM minus
// headroom for the OS and everything that isn't this session — min(max(20%
// of RAM, 8GiB), 24GiB), so small boxes keep breathing room without the
// reserve swallowing them — minus the checkpoint's mapped weight bytes
// (zero-copy weights go resident as decode touches them, so steady state is
// all of them), minus the fixed batch/staging slack (the bounded batch slabs
// + staging headroom — 4GiB flat). ok=false when weights and the fixed
// overhead alone already crowd the box: no room for any KV rows, so the
// caller must fail closed rather than treat a zero/negative budget as room.
func sessionMemoryBudgetBytes(weightsBytes, ramBytes uint64) (budget uint64, ok bool) {
	reserve := min(max(ramBytes/5, 8<<30), 24<<30)
	const fixed = uint64(4) << 30
	if ramBytes <= reserve+weightsBytes+fixed {
		return 0, false
	}
	return ramBytes - reserve - weightsBytes - fixed, true
}

// sessionKVBytesAt returns the KV-cache bytes ONE session of context length
// ctxLen costs for arch: kvRowBytes per row for every cache-owning layer, a
// sliding-window layer's ring capped at SlidingWindow rows once ctxLen
// exceeds the window. This is the forward direction (length → bytes) of the
// same per-row/sliding-window accounting clampContextToRAM inverts (budget →
// length) — the batch-lane admission gate (lane_set.go, admitMemoryBudget)
// uses it to price a lane at the model's already-decided context length
// instead of re-deriving the cost a second way.
func sessionKVBytesAt(arch model.Arch, ctxLen int) uint64 {
	if ctxLen <= 0 {
		return 0
	}
	var total uint64
	for li := range arch.Layer {
		spec := arch.Layer[li]
		if !spec.OwnsCache() {
			continue
		}
		rowBytes := kvRowBytes(spec, arch)
		rows := uint64(ctxLen)
		if arch.SlidingWindow > 0 && arch.SlidingWindow < ctxLen && spec.Attention != model.GlobalAttention {
			rows = uint64(arch.SlidingWindow) // ring-bounded: fixed charge, not per-row
		}
		total += rows * rowBytes
	}
	return total
}

// intFromBytes saturates a byte count to the platform int range. Realistic
// device/RAM sizes stay many orders of magnitude below the ceiling; this only
// guards the theoretical overflow so an absurd value fails a budget.FitsMemory
// check rather than wrapping negative and passing one it shouldn't.
func intFromBytes(b uint64) int {
	if b > math.MaxInt {
		return math.MaxInt
	}
	return int(b)
}

// clampContextToRAM bounds a DEFAULT context so the session's KV plan fits
// the box: the 31B@256K all-defaults run swapped a 96GB machine because the
// window-capped default never consulted RAM. Applied after loadRegistered —
// exact layer geometry and mapped weight bytes in hand — and only when the
// caller left -context unset. The budget model is deliberately conservative:
//   - reserve: headroom for the OS and everything that isn't this session —
//     min(max(20% of RAM, 8GiB), 24GiB), so small boxes keep breathing room
//     without the reserve swallowing them.
//   - weights: the checkpoint's mapped shard bytes (zero-copy weights go
//     resident as decode touches them — steady state is all of them).
//   - fixed: the bounded batch slabs (sdpaPromptS pair runs to its 2×1GiB
//     budget at long context) + staging slack — 4GiB flat.
//   - per row: every cache-owning GLOBAL layer pays K+V at the bf16 rate
//     (2 × kvDim × 2B) — deliberately ignoring the q8 saving, because the
//     q8 GEMM-prefix mirrors transiently return those layers to bf16 size
//     during a long prefill. Sliding owners ring at the window: a fixed,
//     up-front charge instead of a per-row one.
func clampContextToRAM(defCtx int, arch model.Arch, weightsBytes, ramBytes uint64) int {
	if !contextRAMGuardEnabled || defCtx <= 4096 || ramBytes == 0 {
		return defCtx
	}
	budget, ok := sessionMemoryBudgetBytes(weightsBytes, ramBytes)
	if !ok {
		return 4096 // weights alone crowd the box: keep the floor (explicit -context overrides)
	}
	var perRow uint64
	for li := range arch.Layer {
		spec := arch.Layer[li]
		if !spec.OwnsCache() {
			continue
		}
		rowBytes := kvRowBytes(spec, arch)
		if arch.SlidingWindow > 0 && arch.SlidingWindow < defCtx && spec.Attention != model.GlobalAttention {
			cost := uint64(arch.SlidingWindow) * rowBytes // ring-bounded: fixed charge
			if budget <= cost {
				return 4096
			}
			budget -= cost
			continue
		}
		perRow += rowBytes
	}
	if perRow == 0 {
		return defCtx
	}
	rows := budget / perRow
	if rows >= uint64(defCtx) {
		return defCtx
	}
	clamped := max(4096, int(rows)&^1023) // round down to a 1024 boundary
	nativeTraceLog(core.Sprintf(
		"native: default context %d clamped to %d for this box (RAM %dGiB, weights %.1fGiB; explicit -context overrides, LTHN_CONTEXT_RAM_GUARD=0 disables)\n",
		defCtx, clamped, ramBytes>>30, float64(weightsBytes)/(1<<30)))
	return clamped
}

// clampDefaultContextToRAM is the load-path wrapper: sysctl RAM + the mapped
// shard bytes, applied to a default (caller-unset) context only.
func clampDefaultContextToRAM(defCtx int, arch model.Arch, sb *shardBuffers) int {
	return clampContextToRAM(defCtx, arch, sb.totalMappedBytes(), physicalRAMBytes())
}
