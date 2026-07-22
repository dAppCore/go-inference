# Release artifacts — the `lem` Driver/Native grid

`lem` is one binary. The GPU backend is registered at **build** time, and `lem` also
serves its own HTTP API — so a compute box can run a backend-specific `lem` as a remote
GPU **driver**, and a user's local `lem` consumes it over that API. The release surface
therefore ships every per-`{os,arch,backend}` build in **two packagings of the same binary**:

| Packaging | Zip name | Contains (one binary, nothing else) | Role |
|-----------|----------|-------------------------------------|------|
| **Native** | `{os}-{arch}-lem-{backend}-{version}.zip` | `lem` (`lem.exe` on Windows) | Run locally on your machine. |
| **Driver** | `{os}-{arch}-lem-driver-{backend}-{version}.zip` | `lem-{backend}` (`.exe` on Windows) | Deploy on a compute box; serves the GPU over the API. |

Worked example (the maintainer's own setup):

- On his Mac: `macos-aarch64-lem-metal-v0.15.0.zip` → a `lem` binary for mac + metal.
- On his AMD Linux box: `linux-x86_64-lem-driver-amd-v0.15.0.zip` → a `lem-amd` binary
  serving the GPU over the API, which the Mac's `lem` then drives remotely.

**Build once, package twice.** Each cell is built a single time; the one binary is copied
into the Native zip as `lem` and into the Driver zip as `lem-{backend}`. No double build.

## Dimensions

- `os` ∈ {`macos`, `linux`, `windows`}  (GOOS `darwin`→`macos`)
- `arch` ∈ {`x86_64`, `aarch64`}  (GOARCH `amd64`→`x86_64`, `arm64`→`aarch64`)
- `backend` ∈ {`metal`, `amd`, `cuda`, `cpu`}
- `version` = the git tag on a release build; the rolling `dev` prerelease uses `dev-<short-sha>`.

## The grid — valid cells only

| os | arch | backend | Built by | How |
|----|------|---------|----------|-----|
| macos | aarch64 | metal | self-hosted macOS 26 lane (dormant) / manual | `task metallib && task build:embed` |
| macos | aarch64 | cpu | **GitHub** `macos-latest` | native cgo `go build ./cli` |
| macos | x86_64 | cpu | **GitHub** `macos-13` | native cgo |
| linux | x86_64 | amd | **GitLab** homelab | `make lem-amd` |
| linux | x86_64 | cuda | **GitLab** homelab | `make lem-cuda` |
| linux | x86_64 | cpu | **GitHub** `ubuntu-latest` (rolling) · **GitLab** `make lem-cpu-x86` (tag) | native cgo |
| linux | aarch64 | cpu | **GitHub** `ubuntu-24.04-arm` (rolling) · **GitLab** `make lem-cpu-aarch64` (tag) | native cgo |
| windows | x86_64 | cpu | **GitHub** `windows-latest` | native cgo (mingw) |

### Excluded cells (and why)

- **cpu · windows/aarch64** — `duckdb-go-bindings` ships no `windows-arm64` prebuilt lib, so
  the binary cannot link. Not a cell.
- **metal · anything but macos/aarch64** — Metal is Apple-GPU only.
- **amd, cuda · anything but linux/x86_64** — need the ROCm / CUDA toolchain and its Linux libs.
- **any backend with `CGO_ENABLED=0`** — impossible; see below.

## Why every cell is a *native* cgo build (no cross-compile)

`lem` links **DuckDB** (the `go-store` / chathistory driver) through cgo. `duckdb-go-bindings`
guards its prebuilt-lib packages with `//go:build cgo`, so `CGO_ENABLED=0` fails at compile
(`build constraints exclude all Go files ...`). There is **no pure-Go `lem`** and therefore
**no free `GOOS/GOARCH` cross-compile** — cross-building needs a target C toolchain. Each cell
is built natively on a runner of that `os/arch` (GitLab's `lem-cpu-aarch64` is the one
cross-cgo build, via `aarch64-linux-gnu-gcc`). Probed on 2026-07-22 against
`duckdb-go-bindings v0.10504.0`: cgo-off builds fail for every target; native cgo builds pass.

DuckDB's supported set (the prebuilt libs that exist): `darwin/{amd64,arm64}`,
`linux/{amd64,arm64}`, `windows/amd64`. That set defines the valid `cpu` cells.

## Which CI owns which cells

- **GitHub Actions** (`.github/workflows/build.yml`) — the portable **cpu** cells it can build
  natively on hosted runners, on every push to `dev`, published to the rolling `dev`
  prerelease as `dev-<short-sha>` zips (Native + Driver). Hosted runners have no ROCm/CUDA
  toolchain and are older than macOS 26, so they build **only** cpu.
- **GitLab CI** (`.gitlab-ci.yml`, homelab AMD box) — the toolchain-gated **amd** and **cuda**
  cells, plus its own **cpu-x86 / cpu-aarch64**, on a git **tag** via the `release` gate and
  the `Makefile` targets. This side is unchanged (it is green and has the toolchain).
- **Self-hosted Mac lane** — the **metal** cell (see below).

The two systems do not fight: GitHub is the continuous `dev` channel; GitLab is the tagged
release for the Linux/GPU cells.

## The metal cell decision

Metal's Go side (engine/metal, the objc bridge) is no-cgo and cross-compiles, **but** a usable
metal `lem` embeds the MLX + lthn `.metallib` kernels (`-tags embed_metallib`), and those are
compiled by `task metallib` with **Metal 4 / macOS 26** (`-std=metal4.0`,
`CMAKE_OSX_DEPLOYMENT_TARGET=26.0`) and are gitignored (generated, never committed).

- A GitHub **hosted** macOS runner is macOS 14/15 → it **cannot** compile the metallibs. A plain
  (non-embed) build would resolve them at runtime via `MLX_METALLIB_PATH`, but the user has no
  metallib on their machine — so that artifact is **non-runnable**. We do **not** ship a
  fabricated/broken metal binary from hosted CI.
- **Decision:** the metal cell ships from a **self-hosted macOS 26 arm64 runner** running the
  house build path (`task metallib && task build:embed`), producing the self-contained binary.
  The `metal` job in `build.yml` encodes this recipe but is **dormant** — its `if` is false
  unless repo variable `ENABLE_MACOS_METAL=true` is set — so it never queues or fails on a
  normal push. **Interim** (until a self-hosted runner is registered): the maintainer builds
  the metal zip manually on his Mac (the primary dev box already runs `task build:embed` daily).
- **Trade-off:** correctness (a self-contained, runnable metal binary) over convenience (a
  hosted job). The cost is one self-hosted runner or a manual step for the metal cell only.

## Zip layout rule

Each zip contains **exactly one binary and nothing else** — matching what `build.yml` shipped
before (binary only, no licence file). Self-contained cells honour this directly:

- **cpu** — native-cgo `lem`, no GPU engine, no sidecar.
- **metal** — `build:embed` bakes the metallibs in.

**Exception — amd / cuda:** these load a HIP/CUDA kernel **sidecar** (`.hsaco` / `.o`) at
runtime that is not embedded, so those cells ship the binary **plus** its sidecar (the
`Makefile` already tars them together). This is the single documented departure from
binary-only, driven by runtime necessity.

## Not yet wired (follow-ups)

- **Tagged GitHub release.** `build.yml` publishes only the rolling `dev` prerelease (dev
  pushes). Producing tag-versioned (`v*`) zips for the GitHub cells (macos/windows cpu, metal)
  is a small extension: add `push.tags` to `on:` and version from the tag. GitLab already
  handles tagged Linux/GPU cells.
- **GitLab grid-named zips.** GitLab currently uploads the raw `lem-{backend}` binaries (already
  Driver-named) via `build/bin/`. Emitting the `{os}-{arch}-lem-...-{version}.zip` names (with
  the amd/cuda sidecars) is a `Makefile`/`.gitlab-ci.yml` packaging follow-up that must be
  validated on the ROCm box — it cannot be exercised from a macOS lane, so it was left untouched
  here rather than changed blind.
