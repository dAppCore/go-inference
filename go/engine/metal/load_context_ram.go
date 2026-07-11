// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
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
	reserve := min(max(ramBytes/5, 8<<30), 24<<30)
	const fixed = uint64(4) << 30
	if ramBytes <= reserve+weightsBytes+fixed {
		return 4096 // weights alone crowd the box: keep the floor (explicit -context overrides)
	}
	budget := ramBytes - reserve - weightsBytes - fixed
	var perRow uint64
	for li := range arch.Layer {
		spec := arch.Layer[li]
		if !spec.OwnsCache() {
			continue
		}
		lhd, lkv := headDimOf(spec, arch.HeadDim), kvHeadsOf(spec, arch.KVHeads)
		rowBytes := uint64(lkv*lhd) * bf16Size * 2 // K+V per row, bf16 rate
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
