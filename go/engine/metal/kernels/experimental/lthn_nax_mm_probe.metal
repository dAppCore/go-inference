// SPDX-Licence-Identifier: EUPL-1.2

// lthn_nax_mm_probe — PROTOTYPE, NOT WIRED. A viability probe for Metal 4's MXU
// (Matrix eXecution Unit) hardware matmul in go-inference's no-cgo kernel build.
//
// STATUS (2026-07-09): compiles + links into a metallib at -std=metal4.0 (the
// toolchain ceiling — there is NO -std=metal4.1 in MetalToolchain v17.6; Metal 4.x
// *framework* versions do not change the MSL *language* std). It is NOT layout-
// correct yet: the fragment load below is a naive per-lane copy, not the
// get_coord()/metal::tensor load the real tiled GEMM needs — so it will NOT
// produce correct results as-is. This file exists to prove the path is reachable
// and to seed the real work.
//
// WHAT IT PROVES: Metal 4's cooperative-tensor matmul (mpp::tensor_ops::matmul2d,
// the MXU) is usable in our shaders with ZERO host changes — fully inside the
// sovereign no-cgo design. The "std::mdspan trick" is real here, spelled
// metal::tensor (strided views + __metal_load/store_tensor); for matmul it is
// metal::cooperative_tensor driven via MetalPerformancePrimitives (mpp).
//
// SCOPING (the honest bit): the MXU is a MATMUL unit — descriptor tiles M/N/K,
// at least one of which must be 32 when both inputs are cooperative tensors. It
// pays on PREFILL (M = many prompt tokens, a GEMM) and the MTP verify (M = draft
// block). It does NOT help single-token DECODE, which is M=1 matvec (our lthn_qmv,
// bandwidth-bound near-floor) — a 32-wide tile would waste 31/32 of the unit. So
// this is a PREFILL accelerator, not a decode-tok/s lever.
//
// NEXT STEPS to make it real:
//   1. Correct fragment layout — load A/B via metal::tensor + cooperative_tensor
//      load(), or the per-lane get_coord() mapping. Reference: MLX
//      steel/gemm/nax.h (includable here via the build's -I external/mlx).
//   2. Tile + threadgroup K-loop for a full [M x N] output.
//   3. Wire into the prefill GEMM path; A/B deep-context prefill vs the current
//      steel_gemm (31B @16K prefill ~45s is the target pain point).
//
// COMPILE STANDALONE (it is under kernels/experimental/, so metallib:kernels'
// kernels/*.metal glob does NOT build it into the production lib):
//   xcrun -sdk macosx metal -std=metal4.0 -I external/mlx \
//     -c go/engine/metal/kernels/experimental/lthn_nax_mm_probe.metal -o /tmp/x.air
//   xcrun -sdk macosx metallib /tmp/x.air -o /tmp/x.metallib

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void lthn_nax_mm_probe(
    device const half*  A [[buffer(0)]],   // M x K
    device const half*  B [[buffer(1)]],   // K x N
    device       float* C [[buffer(2)]]) { // M x N

  // MXU tile M=16, N=32, K=16 (one dim must be 32). mode::multiply -> C = A @ B.
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, false, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

  auto ct_a = op.get_left_input_cooperative_tensor<half, half, float>();
  auto ct_b = op.get_right_input_cooperative_tensor<half, half, float>();
  auto ct_c = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

  // TODO(layout): get_capacity() is the per-thread fragment size; replace this
  // naive copy with a coordinate-correct load (element -> (row,col) is the
  // simdgroup's map — see nax.h get_coord()).
  for (ushort i = 0; i < ct_a.get_capacity(); i++) ct_a[i] = A[i];
  for (ushort i = 0; i < ct_b.get_capacity(); i++) ct_b[i] = B[i];

  op.run(ct_a, ct_b, ct_c);                // <- the hardware MXU matmul fires here

  for (ushort i = 0; i < ct_c.get_capacity(); i++) C[i] = ct_c[i];
}
