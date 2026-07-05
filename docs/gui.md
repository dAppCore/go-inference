# The LEM desktop app (`gui/`)

`gui/` is the **LEM Desktop** application — a system-tray app for driving and
watching local training and inference. It is a **side app**: its own module,
`dappco.re/go/inference/gui`, distinct from the main `dappco.re/go/inference`
module. It is built on **Wails v3** (`v3.0.0-alpha.71`) and, per its package doc,
ships as a signed native macOS binary (Lethean CIC), a Linux AppImage, and a
Windows installer.

Source: `gui/`. The bridge to go-inference's libraries: `gui/internal/lem/`.

## Builds under `go.work` today

`gui/go.mod` only lists `dappco.re/go`, Wails v3, and `golang.org/x/sys` in its
`require` block, yet the code imports `dappco.re/go/inference/...` and
`dappco.re/go/container`. Those resolve through the workspace (`go.work`, which
`use`s `./gui` alongside `./go` and the `external/` submodules). This is
deliberate: **the external dev branches are not tagged yet**, so the GUI builds
in workspace mode rather than pinning released module versions. Build it from the
repo root with the workspace active.

## Architecture

`main.go` wires five Wails services, sets up the system tray and four windows,
and runs the app with the macOS activation policy set to *accessory* (a
menu-bar/tray app, no dock icon). Configuration is read from the environment
(see below), with sensible fallbacks.

### Services

| Service | File | Role |
|---------|------|------|
| `DashboardService` | `dashboard.go` | reads training + generation metrics, exposes snapshots to the frontend |
| `AgentRunner` | `agent_runner.go` | starts/stops the scoring agent loop |
| `DockerService` | `docker.go` | controls the Docker Compose stack (Forgejo, InfluxDB, inference) |
| `ContainerService` | `container_apple.go` | controls a single Apple container via go-container |
| `TrayService` | `tray.go` | the system tray icon, menu, and aggregate snapshot |

#### DashboardService

Bridges the metrics store for the UI. On startup it runs a refresh loop **every
30 seconds** that queries InfluxDB (through the `lem.InfluxClient` bridge) for:

- `training_status` — per-model run progress (model, run id, status, iteration,
  total iterations, pct)
- `training_loss` — latest train loss per model
- `golden_gen_progress` and `expansion_progress` — dataset-generation progress
- `capability_score` — the model inventory (name, label/tag, accuracy,
  iteration)

`GetSnapshot` returns the assembled `DashboardSnapshot`; `RunQuery` runs an
ad-hoc SQL query against the read-only DuckDB metrics store (`lem.OpenDB`) when a
`LEM_DB` path is configured.

#### AgentRunner

Wraps the scoring agent for desktop use with `Start` / `Stop` / `IsRunning` /
`CurrentTask`. `Start` builds CLI-style args from its config
(`--api-url`, `--influx`, `--influx-db`, `--m3-host`, `--base-model`,
`--work-dir`) and runs `lem.RunAgent(args)` in a background goroutine; that call
blocks until the loop exits. Note: the agent loop does not yet honour context
cancellation, so `Stop` marks the runner stopped but the underlying loop is not
interrupted mid-flight (flagged in the code).

#### DockerService

Manages the LEM Docker Compose stack. `Start`/`Stop` shell out to
`docker compose -f <deploy>/docker-compose.yml up -d` / `down` (via
`golang.org/x/sys/execabs`), with per-service start/stop/restart, `Logs`, and
`Pull`. A status loop **every 15 seconds** parses `docker compose ps --format
json` into per-service `ContainerStatus`.

#### ContainerService

The first slice toward shipping the LEM Runtime GUI on the App Store via **Apple
Containerisation**. It mirrors `DockerService`'s shape but is backed by
go-container's `AppleProvider` (the `container` CLI shipped with macOS 26+)
rather than `docker compose`, and it runs **alongside** DockerService rather than
replacing it. `Start` launches a detached OCI container (default image
`docker.io/library/alpine:latest`), with `Stop`/`Restart`/`Logs`/`GetStatus` and
a 15-second status loop that prefers the provider's tracked set and falls back to
a CLI inspect.

#### TrayService

Owns the system tray. `GetSnapshot` aggregates the other services into a
`TraySnapshot` (stack running, contained running + status, agent running + task,
training rows, generation stats, models, docker service count). The tray menu
offers: Start/Stop Services (Docker stack), Start/Stop Contained Service,
Start/Stop Scoring Agent, Open Dashboard / Workbench / Forge (opens
`http://localhost:3000` in the browser), a Training submenu, Settings, and Quit.

### Windows and frontend

`main.go`/`tray.go` register four Wails webview windows, all dark
(`RGB(15,23,42)`), served from the embedded `gui/frontend/` SPA:

| Window | Route | Notes |
|--------|-------|-------|
| `tray-panel` | `/tray` | frameless dropdown attached to the tray icon |
| `dashboard` | `/dashboard` | shown on first launch |
| `workbench` | `/workbench` | model scoring, probes |
| `settings` | `/settings` | |

The frontend is a single `gui/frontend/index.html` served with an SPA fallback
(`spaHandler`): any unknown path rewrites to `/`, so the routes above are
client-side.

## The `gui/internal/lem` bridge

`gui/internal/lem/lem.go` is the shim between the desktop GUI and go-inference's
consolidated packages. It is the new home of what the GUI used to import from
`dappco.re/lthn/lem/pkg/lem` before the AI features consolidated into
go-inference:

- the metrics client + DuckDB store now live in `dappco.re/go/inference/eval/datapipe`,
- the scoring agent loop in `dappco.re/go/inference/agent`.

The shim keeps the GUI's call sites (`lem.NewInfluxClient`, `lem.OpenDB`,
`lem.InfluxClient`, `lem.DB`, `lem.RunAgent`) intact, and adapts datapipe's
`core.Result` returns into the `(value, error)` pairs the dashboard code expects.

## Configuration (environment)

| Variable | Default | Used for |
|----------|---------|----------|
| `INFLUX_URL` | `http://localhost:8181` | metrics source |
| `INFLUX_DB` | `training` | metrics database |
| `LEM_API_URL` | `http://localhost:8080` | inference/API endpoint for the agent |
| `M3_HOST` | `10.69.69.108` | remote MLX host for the scoring agent |
| `BASE_MODEL` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | agent base model |
| `LEM_DB` | `""` | DuckDB metrics store path (enables `RunQuery`) |
| `WORK_DIR` | `<tmp>/scoring-agent` | scoring-agent work directory |
| `LEM_DEPLOY_DIR` | auto-detected | directory holding `docker-compose.yml` |
| `LEM_CONTAINER_NAME` | `lem-contained` | Apple container name |
| `LEM_CONTAINER_IMAGE` | (unset → `alpine:latest`) | Apple container image |

`LEM_DEPLOY_DIR` is auto-located by `findDeployDir`: it looks for a `deploy/`
directory containing `docker-compose.yml` next to the executable, then relative
to the working directory, falling back to the literal `deploy`. That deploy tree
(the Compose stack) is not part of this repository, so the Docker features expect
it to be provided at deploy time.

## Building

Build with the Wails v3 tooling from the repo root under the active workspace.
Because the external dependencies resolve through `go.work` (not tagged module
versions), do not build `gui/` in isolation with `GOWORK=off` until the external
dev branches are tagged.
