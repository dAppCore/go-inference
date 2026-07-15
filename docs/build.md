# Building `lem` and the Metal kernel libraries

Since the go-mlx / go-rocm retirement, go-inference owns the **whole Metal build
chain**. This document covers the `Taskfile.yml` targets that build the two GPU
kernel libraries and the `lem` binary — including the self-contained embed build
that makes "you only need go-inference" literally true.

Everything here is the Apple/Metal path. The AMD `engine/hip` engine (which does
carry cgo) is built from this same repo on the AMD/linux box; it is out of scope
for this document.

## Prerequisites

- **macOS 26 or later** — the build targets deployment target `26.0`
  (`-mmacosx-version-min=26.0`, `CMAKE_OSX_DEPLOYMENT_TARGET=26.0`).
- **Xcode command-line tools** — `xcrun metal` / `xcrun metallib` compile the
  fused kernels; the Metal toolchain must be present.
- **CMake** — builds Apple MLX's kernels.
- **Task** (`go-task`) — the runner for `Taskfile.yml`.
- **Go 1.26**.
- The `external/mlx` submodule initialised (`git submodule update --init
  external/mlx`).

## The two kernel libraries

The Apple engine dispatches two compiled Metal libraries, both **built from
source in this repo** — there is no go-mlx dependency:

| Library | Built from | Contents |
|---------|-----------|----------|
| `mlx.metallib` | Apple MLX (`external/mlx`) + the lthn patches, via CMake | Apple MLX's own kernels: `steel_gemm`, `affine_qmv`, `vv_*`, rms, rope, sdpa |
| `lthn_kernels.metallib` | `go/engine/metal/kernels/*.metal`, via `xcrun` | go-inference's own fused kernels (39 `.metal` sources: the FFN/attention/layer megakernels, gelu-gate-mul, qgemv, rmsnorm-residual, sdpa variants, MoE router, …) |

At runtime the engine loads `mlx.metallib` (named by `MLX_METALLIB_PATH`) and
then looks for `lthn_kernels.metallib` **as a sibling in the same directory**.
The sibling is optional: if it is absent, the fused ops fall back to composed
primitives.

## Patch-not-vendor: `external/mlx` + `patches/mlx/`

`external/mlx` is Apple's canonical MLX (`github.com/ml-explore/mlx`) as a git
submodule, **pinned at v0.32.0**. Rather than fork or vendor a modified MLX, the
10 lthn patches in `patches/mlx/` are applied **on top at build time** and then
reverted, so the submodule stays pristine in `git status`.

The patch set (`patches/mlx/0001…0010`):

- `0001` — `MLX_METALLIB_PATH` override (defensive metallib resolution)
- `0002` — unbound threads adopt the process canonical pool
- `0003` — env-gated compile-cache decision trace
- `0004`–`0010` — the decode-replay perf line: a command recorder that captures
  the flat Metal encode, the replay primitive with a finalize barrier,
  buffer-pin free deferral, a captured-payload byte hash (proves no divergence),
  step-level capture, and the end-to-end programmatic replay.

To pull upstream MLX updates: **bump the submodule pin, then rebase the patch
set** on the new tag. Nothing is vendored, so tracking Apple MLX stays a
pin-bump-plus-rebase, not a merge.

## `task metallib` — build both libraries

```bash
task metallib          # runs metallib:mlx then metallib:kernels
```

### `task metallib:mlx`

Starts from the pristine pinned MLX, applies every `patches/mlx/*.patch` with
`git apply`, configures CMake (`MLX_BUILD_METAL=ON`; tests, benchmarks, examples
and Python bindings all off; `CMAKE_OSX_DEPLOYMENT_TARGET=26.0`), builds the
`mlx` target in parallel, copies the compiled `mlx.metallib` out, and restores
`external/mlx` to pristine (`git checkout` + `git clean`) so the submodule is
clean again.

### `task metallib:kernels`

Compiles each `go/engine/metal/kernels/*.metal` to a `.air` object with
`xcrun -sdk macosx metal -std=metal4.0 -I external/mlx` (the MLX headers are on
the include path), then links them with `xcrun -sdk macosx metallib` into
`lthn_kernels.metallib`.

### Output paths

Both libraries land under `build/dist/lib/`: `mlx.metallib` (copied out of the
CMake build) and `lthn_kernels.metallib` (linked from the compiled `.air`
objects). The Taskfile's `MLX_METALLIB_PATH` env points at
`build/dist/lib/mlx.metallib`, and the embed build reads both from that same
directory (see below).

## `task build` — the external-metallib binary

```bash
task metallib          # once, to produce the metallibs
task build             # -> bin/lem
```

Builds `bin/lem` with `-tags metal_runtime -trimpath` and the darwin ldflags
(`-extldflags=-mmacosx-version-min=26.0`), using a dedicated build cache under
`/private/tmp/lem-dev/gocache`. This binary resolves its metallibs **externally**
at runtime via `MLX_METALLIB_PATH` (and the sibling lookup for
`lthn_kernels.metallib`), so `task metallib` must have run first.

## `task build:embed` — the self-contained binary

```bash
task metallib          # once
task build:embed       # -> bin/lem, SELF-CONTAINED
```

This is the "you only need go-inference" build. It:

1. Checks `build/dist/lib/{mlx,lthn_kernels}.metallib` exist (errors telling you
   to run `task metallib` if not).
2. `gzip -9`s both into `cli/{mlx,lthn_kernels}.metallib.gz` next to
   `embed_metallib.go`.
3. Builds `bin/lem` with `-tags "metal_runtime embed_metallib"`.

Under `-tags embed_metallib`, `cli/embed_metallib.go` is compiled in and
`//go:embed`s both gzipped libraries into the binary. At process start (before
any Metal device init) its `init()`:

- Skips entirely if the operator already set `MLX_METALLIB_PATH` — an explicit
  path always outranks the embedded copy (the same set-if-unset contract the
  engine honours).
- Otherwise gunzips both libraries into a single **content-addressed** temp dir
  (`os.TempDir()/lthn-lem/<sha256(mlx+kernels)[:8]>`), so a version bump lands in
  a fresh dir and the two libraries always match. Both extract into the one dir
  so the engine's sibling lookup finds `lthn_kernels.metallib` beside
  `mlx.metallib`.
- Sets `MLX_METALLIB_PATH` at the extracted `mlx.metallib`.

Extraction is idempotent (a present non-empty file is trusted) and writes via a
temp sibling + rename so a concurrent start never sees a half-written file. Any
failure is best-effort: it leaves `MLX_METALLIB_PATH` unset so the engine falls
back to normal external resolution rather than crashing at import time.

The result runs **from any path** with nothing external to ship or resolve — the
single-artifact USP. The trade-off is size: the embedded `mlx.metallib.gz` alone
is ~47 MB, so the embed tag is deliberately kept out of routine `go build` /
`go test` / CI runs (without the tag, `embed_metallib.go` is excluded and the
engine resolves the metallib externally).

## Runtime resolution recap

| Build | How the metallib is found |
|-------|---------------------------|
| plain `go build` / `go test` | not embedded; `MLX_METALLIB_PATH` or a colocated `mlx.metallib`; `lthn_kernels.metallib` looked up as a sibling |
| `task build` (`metal_runtime`) | same external resolution; `task metallib` must have produced the libraries |
| `task build:embed` (`metal_runtime embed_metallib`) | libraries baked in, extracted to a content-addressed temp dir, `MLX_METALLIB_PATH` set before Metal init unless the operator set it first |

The engine-side resolution (env var name `MLX_METALLIB_PATH`, the sibling
`lthn_kernels.metallib` lookup, the composed-primitive fallback) lives in
`go/engine/metal/device.go`.
