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

go-inference has no Taskfile yet, so the build is documented here rather than
ported as a task (no new build infrastructure invented). The pipeline compiles
every `*.metal` in this directory to an `.air` and links them into one
`metallib` — the same steps go-mlx's `task build:kernels` runs, retargeted at
this directory:

```sh
# Run from the module root (go/). MLX_HEADERS points at the mlx headers the
# kernels #include (e.g. the go-mlx checkout's lib/mlx); OUT is the dist dir
# the engine will find via MLX_METALLIB_PATH's sibling lookup.
MLX_HEADERS=../../../go-mlx/lib/mlx
OUT=dist/lib
mkdir -p "$OUT"
airs=""
for m in engine/metal/kernels/*.metal; do
  air="/tmp/$(basename "${m%.metal}").air"
  xcrun -sdk macosx metal -std=metal4.0 -I "$MLX_HEADERS" -c "$m" -o "$air"
  airs="$airs $air"
done
xcrun -sdk macosx metallib $airs -o "$OUT/lthn_kernels.metallib"
```

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
