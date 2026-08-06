<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# engine/metal custom kernels

The `*.metal` sources here are the native/metal engine's **own** fused compute
kernels (fused gelu-gate, rmsnorm-residual, qgemv, sdpa variants, the layer /
ffn / attn megakernels, …) — the kernels the stock `mlx.metallib` does not
ship. `device.go` loads the compiled result, `lthn_kernels.metallib`, as a
sibling of the main metallib named by `MLX_METALLIB_PATH` (see
`siblingMetallib`). When it is absent those ops fall back to composed
primitives, so a checkout without the metallib still builds and runs.

## Layout

The tree follows the QuixiCore family taxonomy — **semantic family first, file
per operation** — the same vocabulary `engine/hip/kernels/` uses, so the two
engines read alike:

```text
kernels/
  activations/    elementwise gates + vector primitives: gelu/silu gate-multiply,
                  vector·scalar, broadcast row multiply, contiguous copy
  attention/      the sequence mixers: SDPA vector/paged/multi-query/ring/runtime-dim,
                  the steel flash instantiations (256 / split-D 512 / q8 / windowed),
                  the composed lane's attention core, device-KV V-append, causal
                  score-slab softmax, and the gated-delta recurrence
  embedding/      embedding gather + dequantise, and the gemma PLE slab family
  megakernel/     whole-stage single-dispatch kernels separated by in-kernel
                  device-scope grid barriers (attn / ffn / full layer / the gemv2
                  foundation)
  moe/            routing, top-k, grouped gather/scatter/sort, expert combine, and
                  the affine expert-route projections (gather-qmv, gelu-qmv)
  norms/          RMSNorm family, residual-add, QK-norm+RoPE, MoE combine
  probes/         device-capability probes compiled into the library and read by
                  tests: grid-sync feasibility, cross-threadgroup coherency
  quant/          MLX affine quantised projection (qgemv, qmv rows, fused rms+qmv,
                  fused V-projection) and the TurboQuant KV codec + its readers
  sampling/       LM-head argmax / top-k / sample tiles
  training/       fused softmax cross-entropy forward+backward
  experimental/   NOT compiled into the library — prototypes seeding future work
```

Within a family, `<operation>.metal` holds the kernels and their local device
helpers; a `<operation>_impl.h` holds a body shared verbatim by two kernels in
the **same** family (`moe/lthn_gelu_qmv_impl.h`, `moe/lthn_router_topk_impl.h`).
Those headers are included with a quoted, unqualified `#include`, so a shared
body and its consumers must live in one directory — if a new sharer arrives from
another family, move the shared body to `common/` and add the include path to
the Taskfile rather than reaching across families with `../`.

**Why the families differ from `engine/hip`'s in two places.** `engine/hip`
ships `contract/` (its launch-ABI translation unit) and `lora/`; neither exists
here — Metal kernels carry their ABI in their own `[[buffer(n)]]` attributes and
there is no LoRA kernel on this engine. This engine adds `megakernel/` and
`probes/`, which HIP has no counterpart for: the grid-barrier megakernel line is
an Apple-Silicon-specific lever, and its two feasibility probes ship inside the
library because the tests dispatch them. `common/` and `matmul/` are absent by
observation, not by policy — there is no engine-wide numeric header, and every
projection kernel here is quantised, so they land in `quant/`.

**Why no `variants/` tree.** As on `engine/hip`: the axis that is real here is
head dim, quantisation scheme and tile shape, and it is carried in the kernel
symbol name (`lthn_attn_q8_bfloat16_bq32_bk16_bdh256_nh1_wm4_wn1`) or in a
function constant, not in a directory.

## Adding a kernel

1. Pick the family by **what the kernel does**, not by its filename prefix —
   `norms/lthn_moe_combine_norms.metal` is the MoE block's norm/combine tail and
   sits in `norms/` exactly as `rocm_moe_combine` does on `engine/hip`.
2. Put the kernel body, and any device helper only it uses, in
   `<family>/<operation>.metal`. Add a new file rather than growing an unrelated
   one — this tree is file-per-operation on purpose.
3. If two kernels must share a body byte-for-byte, put it in
   `<family>/<operation>_impl.h` beside them (see the include note above).
4. **No manifest to edit.** Unlike `engine/hip`'s single amalgamation TU, every
   `.metal` file here compiles to its own `.air` and the linker joins them, so
   ordering carries no meaning and the build discovers sources by recursive
   glob. A new family directory is picked up with no build change; only
   `experimental/` is excluded.
5. Add the Go driver's `_test.go` gate. The engine resolves kernels by function
   name at runtime, so a kernel with no named pipeline lookup is dead weight.

## Building `lthn_kernels.metallib`

The repo Taskfile owns the build — from the repo root:

```sh
task metallib:kernels   # this tree's *.metal → build/dist/lib/lthn_kernels.metallib
task metallib           # both libraries (MLX's mlx.metallib + ours)
```

The pipeline compiles every `*.metal` under `kernels/` (recursively, minus
`experimental/`) to an `.air` and links them into one `metallib`; headers come
from the pinned `external/mlx` submodule. `-fno-fast-math` matches MLX's own
kernel build — the byte-identity twins must not let the compiler reassociate fp
chains the oracle kernels keep in source order.

To read what actually shipped in the library — the parity instrument for any
kernel move or rename:

```sh
xcrun metal-nm build/dist/lib/lthn_kernels.metallib | awk '{print $3}' | sort
```

There is no hand-maintained symbol list in this file on purpose: the library
exports 200+ names, most of them template instantiations, and a copy would rot
within a week. `metal-nm` is the receipt.

## Profiling a kernel in flight

The engine ships a one-shot programmatic GPU capture (`gpu_capture.go`) — the
per-dispatch occupancy / limiter / per-line view no Go-side clock can see:

```sh
task capture:serve MODEL=<snapshot dir>   # boots a capture-armed serve (foreground)
# warm it with a request, then in another terminal:
task capture:fire                          # the NEXT request's round → ~/Desktop/round.gputrace
```

Open the `.gputrace` in Xcode (Metal Debugger). One capture per serve process;
restart `capture:serve` to take another.

## Running the engine + its tests

The engine resolves both metallibs from one env var — the custom kernels are
found beside the main metallib:

```sh
export MLX_METALLIB_PATH=/path/to/build/dist/lib/mlx.metallib   # lthn_kernels.metallib sits beside it
go test -tags metal_runtime ./engine/metal/... -count=1
```

`task metallib && task test:metal` from the repo root does both in the layout
the Taskfile already sets up.
