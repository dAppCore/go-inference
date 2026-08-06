// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

// lthn_softmax_xent_rows_f32 — fused softmax cross-entropy FORWARD + BACKWARD over one training
// row's logits, one threadgroup per row. It is the device twin of native.CrossEntropyBackwardF32
// (train_optim.go): the whole [rows × vocab] softmax backward that on the host is a serial per-row
// walk of ~2·vocab transcendentals (the 91%-of-step tax the batched capture forward left behind).
//
// Per row r (this threadgroup):
//   m      = max_i logits[r,i]                                   // sweep 1
//   sum    = Σ_i exp(logits[r,i] − m)                            // sweep 2 (tree-reduced)
//   rowLoss[r] = (log(sum) + m) − logits[r, target[r]]          // raw NLL; the host sums ×invRows
//   dLogits[r,i] = (softmax(logits[r])_i − [i==target]) · invRows // sweep 3
//
// The kernel deliberately leaves the mean scaling of the loss to the host (rowLoss holds the raw
// per-row NLL) so the loss reduction stays in the host's f64 accumulator — byte-for-byte the shape
// CrossEntropyBackwardF32 already uses (lossSum f64, then ×inv). The gradient's ×invRows is folded
// in here because it is per-element and cheap. Maths is f32 throughout; the tree reduction over vocab
// (simd_sum + a 32-lane threadgroup pass) keeps the sum's relative error at ~log2(vocab)·2⁻²³ ≈ 1e-6,
// well inside the 1e-5 tolerance the host oracle is checked against (train_xent_device_test.go). exp
// is the precise variant (not fast::exp) — the f32 head logits are the trainer's real precision and
// the loss trajectory must match the host reference, so a 2⁻¹⁴ fast-exp error is not affordable here.
//
// One threadgroup per row, 1024 threads striding the vocab (query s reads logits[s], s+1024, …), so a
// 262144-vocab row is 256 elements/thread — occupancy-bound, no grid barrier, no per-row storage.
kernel void lthn_softmax_xent_rows_f32(
    device const float* logits  [[buffer(0)]], // [rows, vocab] row-major
    device const int*   targets [[buffer(1)]], // [rows]
    device float*       dLogits [[buffer(2)]], // [rows, vocab] out
    device float*       rowLoss [[buffer(3)]], // [rows] out (raw per-row NLL; host sums ×invRows)
    const constant int&   vocab   [[buffer(4)]],
    const constant float& invRows [[buffer(5)]], // 1/rows, matching the host mean scaling
    uint row      [[threadgroup_position_in_grid]],
    uint lid      [[thread_position_in_threadgroup]],
    uint lsize    [[threads_per_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  device const float* lr = logits + size_t(row) * size_t(vocab);
  device float* dr = dLogits + size_t(row) * size_t(vocab);
  const int V = vocab;
  const int simds = int(lsize) / 32;
  const int tgt = targets[row];

  threadgroup float tgRed[32]; // per-simd partials, reused across the two reductions
  threadgroup float tgShared[2]; // [0] = row max, [1] = row sum, broadcast to every thread

  // sweep 1: row max
  float m = -MAXFLOAT;
  for (int i = int(lid); i < V; i += int(lsize)) {
    m = max(m, lr[i]);
  }
  m = simd_max(m);
  if (simd_lid == 0) {
    tgRed[simd_gid] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = (int(lid) < simds) ? tgRed[lid] : -MAXFLOAT;
  m = simd_max(m);
  if (lid == 0) {
    tgShared[0] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = tgShared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup); // frees tgRed for the sum reduction

  // sweep 2: sum of exp(logit − max), tree-reduced
  float sum = 0.0f;
  for (int i = int(lid); i < V; i += int(lsize)) {
    sum += exp(lr[i] - m);
  }
  sum = simd_sum(sum);
  if (simd_lid == 0) {
    tgRed[simd_gid] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sum = (int(lid) < simds) ? tgRed[lid] : 0.0f;
  sum = simd_sum(sum);
  if (lid == 0) {
    tgShared[1] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sum = tgShared[1];

  // loss: raw per-row NLL = logSumExp − logit[target]; the host multiplies the summed rows by 1/rows.
  if (lid == 0) {
    rowLoss[row] = (log(sum) + m) - lr[tgt];
  }

  // sweep 3: gradient dr[i] = (softmax_i − onehot_i) · invRows
  const float inv = 1.0f / sum;
  for (int i = int(lid); i < V; i += int(lsize)) {
    float p = exp(lr[i] - m) * inv;
    float g = (i == tgt) ? (p - 1.0f) : p;
    dr[i] = g * invRows;
  }
}
