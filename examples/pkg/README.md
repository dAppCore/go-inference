<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Building your app against go-inference — the backend matrix

Every example here (and any app you build the same way) selects its GPU
engine with one blank import:

```go
import _ "dappco.re/go/inference/examples/internal/engine"
```

Copy that tiny package into your own project (it is three build-tagged
files) and the platform picks the engine: **darwin/arm64 → metal**,
**linux/amd64 → hip** (which itself carries the AMD, CUDA and CPU lanes).
`inference.LoadModel` then resolves the first available backend — a macOS
app + go-inference pairing needs nothing else; it simply defaults to metal.

The conventional binary naming is one build per backend:

```
yourapp-mlx   darwin/arm64  metal engine        (Apple silicon)
yourapp-amd   linux/amd64   hip engine, ROCm    (AMD GPUs)
yourapp-cuda  linux/amd64   hip engine, nvidia  (CUDA GPUs)
yourapp-cpu   linux         hip engine, cpu     (no GPU, fully static)
```

`examples/Makefile` builds all the examples in exactly these variants —
use it as the template for your own app's build.

## Apple (mlx) — Taskfile territory

The Apple build is plain Go; the only artefact is the Metal shader library,
which the repo Taskfile owns:

```sh
task metallib                 # once, repo root: build/dist/lib/{mlx,lthn_kernels}.metallib
go build -o yourapp-mlx .     # no tags, no cgo
MLX_METALLIB_PATH=<repo>/build/dist/lib/mlx.metallib ./yourapp-mlx
```

(The `lem` release binary embeds both metallibs via `-tags embed_metallib`;
for your own app the env var is the simple path.)

## AMD / CUDA / CPU — Makefile territory

The linux lanes are owned by the **repo-root Makefile**, which builds the
static HIP archives and the per-target kernel sidecars the binary loads at
runtime. The reference targets — `make help` at the repo root lists them:

| target | what it builds |
|--------|----------------|
| `make lthn-amd` | ROCm binary (cgo + static HIP archives) + HSACO kernel sidecar |
| `make lthn-cuda` | HIP/CUDA binary + CUDA kernel sidecar |
| `make lthn-cpu-x86` / `lthn-cpu-aarch64` | fully static CPU binaries + CPU kernel sidecars |

To build **your own app** for those lanes, mirror what those targets do:

```sh
# AMD (ROCm) — on a linux/amd64 host with the ROCm LLVM toolchain:
make hsa-static-archive hip-static-archive hip-amd   # once, repo root
make hip-link-info                                   # prints the CGO_LDFLAGS to use
CGO_ENABLED=1 CGO_LDFLAGS="<from hip-link-info>" \
  go build -tags rocm_static_hip -o yourapp-amd .
# ship the HSACO kernel sidecar the root Makefile placed in build/bin/ beside your binary

# CUDA — same shape with the hip-nvidia kernel: make hip-nvidia, then as above.

# CPU — no cgo, cross-compilable from anywhere:
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o yourapp-cpu .
# pair with the CPU kernel sidecar from: make hip-cpu-x86_64
```

The release binaries are dependency-guarded (no shared ROCm libraries, no
RPATH) — `make release-dependency-guard` is the check; hold your own app to
the same bar if you redistribute it.

## Defaulting vs pinning

One binary can also carry several backends: registration is additive and
`inference.Default()` prefers metal → rocm → llama_cpp → first available.
Pin explicitly when you need to:

```go
r := inference.LoadModel(path, inference.WithBackend("rocm"))
```

`examples/pkg/backends` prints the live registry — including the honest
"registered but unavailable on this platform" stubs — so you can see what a
given build of your app actually carries.
