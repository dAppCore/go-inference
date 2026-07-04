// SPDX-Licence-Identifier: EUPL-1.2

// lthn_coherency_probe — does Metal give reliable cross-DISTANT-threadgroup producer→consumer data
// visibility? The megakernel's event-driven handoff needs it, and Metal has no release/acquire ordering
// (only memory_order_relaxed) — BUT atomics are L2-coherent. This probes two handoffs side by side over a
// grid barrier: each TG writes its tag to slot[tgid], then ALL TGs sync, then TG 0 reads EVERY slot
// (so it reads distant TGs it never overlapped). PLAIN writes/reads can sit stale in per-TG L1; ATOMIC
// writes/reads go through L2. If atomic reads all tags correctly where plain doesn't, the megakernel's
// cross-TG dependency IS expressible on Metal (move handoff data through atomics). One thread per TG does
// the slot work; TG 0 does the verification. numTG threadgroups, any threadsPerTG.
#include <metal_stdlib>
using namespace metal;

static inline void grid_barrier(device atomic_uint* arrive, uint numTG, uint tid, uint maxSpin) {
  // macOS 26 / metal3.2: DEVICE-scope barrier + seq_cst device fence give cross-threadgroup ordering —
  // the primitive metal3.1 lacked (default threadgroup_barrier only orders within a TG, hence stale reads).
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device, thread_scope_device);
  if (tid == 0u) {
    atomic_fetch_add_explicit(arrive, 1u, memory_order_relaxed);
    for (uint i = 0u; i < maxSpin; i++) {
      if (atomic_load_explicit(arrive, memory_order_relaxed) >= numTG) break;
    }
  }
  threadgroup_barrier(mem_flags::mem_device, thread_scope_device);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}

[[kernel]] void lthn_coherency_probe(
    device uint*        plain   [[buffer(0)]],  // [numTG] plain cross-TG slots
    device atomic_uint* atom    [[buffer(1)]],  // [numTG] atomic cross-TG slots
    device atomic_uint* arrive  [[buffer(2)]],  // grid barrier counter
    device uint*        result  [[buffer(3)]],  // result[0]=plain-ok-count, result[1]=atomic-ok-count
    const constant uint& numTG  [[buffer(4)]],
    const constant uint& maxSpin [[buffer(5)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  const uint tag = tgid + 100u;
  if (tid == 0u) {
    plain[tgid] = tag;                                              // plain write (may stay in L1)
    atomic_store_explicit(&atom[tgid], tag, memory_order_relaxed);  // atomic write (→ L2)
  }
  grid_barrier(arrive, numTG, tid, maxSpin);
  // TG 0 reads EVERY slot — including distant TGs it never co-resided with.
  if (tgid == 0u && tid == 0u) {
    uint plainOK = 0u, atomOK = 0u;
    for (uint i = 0u; i < numTG; i++) {
      if (plain[i] == i + 100u) plainOK++;
      if (atomic_load_explicit(&atom[i], memory_order_relaxed) == i + 100u) atomOK++;
    }
    result[0] = plainOK;
    result[1] = atomOK;
  }
}
