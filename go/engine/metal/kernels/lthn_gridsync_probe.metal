// SPDX-Licence-Identifier: EUPL-1.2

// lthn_gridsync_probe — feasibility probe for a device-wide GRID BARRIER on Apple Silicon: the primitive a
// full-layer decode megakernel needs (all threadgroups finish stage 1, sync, then stage 2 — replacing the
// per-op ICB SetBarriers with ONE in-kernel sync). Metal does NOT guarantee threadgroup co-residency, so if
// `numTG` exceeds what the GPU can co-schedule, an unbounded atomic spin would DEADLOCK. This uses a BOUNDED
// spin so a would-be-deadlock is detected (not hung): each threadgroup's leader arrives, then spins up to
// maxSpin for the counter to reach numTG. out[tg] records the counter value it saw — if every entry == numTG
// the grid barrier completed (all TGs co-resident); any entry < numTG means the GPU could not co-schedule
// them (the megakernel approach would deadlock at that TG count).
#include <metal_stdlib>
using namespace metal;

[[kernel]] void lthn_gridsync_probe(
    device atomic_uint* counter [[buffer(0)]],
    device uint*        out     [[buffer(1)]],
    const constant uint& numTG  [[buffer(2)]],
    const constant uint& maxSpin [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]]) {
  threadgroup uint seen;
  if (lid == 0) {
    atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    uint c = 0;
    for (uint i = 0; i < maxSpin; i++) {
      c = atomic_load_explicit(counter, memory_order_relaxed);
      if (c >= numTG) {
        break;
      }
    }
    seen = c;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {
    out[tgid] = seen; // == numTG ⇒ this TG saw the barrier complete
  }
}
