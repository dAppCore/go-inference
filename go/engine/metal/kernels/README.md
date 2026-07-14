<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# engine/metal custom kernels

The `*.metal` sources here are the native/metal engine's **own** fused compute
kernels (fused gelu-gate, rmsnorm-residual, qgemv, sdpa variants, the layer /
ffn / attn megakernels, …) — the kernels the stock `mlx.metallib` does not
ship. `device.go` loads the compiled result, `lthn_kernels.metallib`, as a
sibling of the main metallib named by `MLX_METALLIB_PATH` (see
`siblingMetallib`). When it is absent those ops fall back to composed
primitives, so a checkout without the metallib still builds and runs.

## Building `lthn_kernels.metallib`

The repo Taskfile owns the build — from the repo root:

```sh
task metallib:kernels   # this directory's *.metal → build/dist/lib/lthn_kernels.metallib
task metallib           # both libraries (MLX's mlx.metallib + ours)
```

The pipeline compiles every `*.metal` here to an `.air` and links them into
one `metallib` (headers come from the pinned `external/mlx` submodule).

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
export MLX_METALLIB_PATH=/path/to/dist/lib/mlx.metallib   # lthn_kernels.metallib sits beside it
go test -tags metal_runtime ./engine/metal/... -count=1
```

Today both live in the go-mlx checkout's `dist/lib/`
(`mlx.metallib` + `lthn_kernels.metallib`); point `MLX_METALLIB_PATH` there.
When go-inference grows a Taskfile, add a `build:kernels` task that runs the
pipeline above so the engine repo can produce its own metallib.
