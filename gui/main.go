// Package main provides the LEM Desktop application.
// A system tray app inspired by BugSETI that bundles:
// - Local Forgejo for agentic git workflows
// - InfluxDB for metrics and coordination
// - Inference proxy to M3 MLX or local vLLM
// - Scoring agent for automated checkpoint evaluation
// - Lab dashboard for training and generation monitoring
//
// Built on Wails v3 — ships as a signed native binary on macOS (Lethean CIC),
// Linux AppImage, and Windows installer.
package main

import (
	"embed"
	"io/fs"
	"net/http"

	core "dappco.re/go"
	"dappco.re/go/inference/gui/icons"
	"github.com/wailsapp/wails/v3/pkg/application"
	"github.com/wailsapp/wails/v3/pkg/events"
)

//go:embed all:frontend
var assets embed.FS

// Tray icon data — placeholders until real icons are generated.
var (
	trayIconTemplate = icons.Placeholder()
	trayIconLight    = icons.Placeholder()
	trayIconDark     = icons.Placeholder()
)

func main() {
	// Strip embed prefix so files serve from root.
	staticAssets, err := fs.Sub(assets, "frontend")
	if err != nil {
		core.Print(core.Stderr(), "%v\n", err)
		core.Exit(1)
	}

	// ── Configuration ──
	influxURL := envOr("INFLUX_URL", "http://localhost:8181")
	influxDB := envOr("INFLUX_DB", "training")
	apiURL := envOr("LEM_API_URL", "http://localhost:8080")
	m3Host := envOr("M3_HOST", "10.69.69.108")
	baseModel := envOr("BASE_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
	dbPath := envOr("LEM_DB", "")
	workDir := envOr("WORK_DIR", core.PathJoin(core.TempDir(), "scoring-agent"))
	deployDir := envOr("LEM_DEPLOY_DIR", findDeployDir())

	// ── Services ──
	dashboardService := NewDashboardService(influxURL, influxDB, dbPath)
	dockerService := NewDockerService(deployDir)
	containerService := NewContainerService(envOr("LEM_CONTAINER_NAME", "lem-contained"), envOr("LEM_CONTAINER_IMAGE", ""))
	agentRunner := NewAgentRunner(apiURL, influxURL, influxDB, m3Host, baseModel, workDir)
	trayService := NewTrayService(nil)

	services := []application.Service{
		application.NewService(dashboardService),
		application.NewService(dockerService),
		application.NewService(containerService),
		application.NewService(agentRunner),
		application.NewService(trayService),
	}

	// ── Application ──
	app := application.New(application.Options{
		Name:        "LEM",
		Description: "Lethean Ethics Model — Training, Scoring & Inference",
		Services:    services,
		Assets: application.AssetOptions{
			Handler: spaHandler(staticAssets),
		},
		Mac: application.MacOptions{
			ActivationPolicy: application.ActivationPolicyAccessory,
		},
	})

	// Wire up references.
	trayService.app = app
	trayService.SetServices(dashboardService, dockerService, containerService, agentRunner)

	// Set up system tray.
	setupSystemTray(app, trayService, dashboardService, dockerService, containerService)

	// Show dashboard on first launch.
	app.Event.RegisterApplicationEventHook(events.Common.ApplicationStarted, func(event *application.ApplicationEvent) {
		if w, ok := app.Window.Get("dashboard"); ok {
			w.Show()
			w.Focus()
		}
	})

	core.Print(core.Stderr(), "Starting LEM Desktop...\n")
	core.Print(core.Stderr(), "  - System tray active\n")
	core.Print(core.Stderr(), "  - Dashboard ready\n")
	core.Print(core.Stderr(), "  - InfluxDB: %s/%s\n", influxURL, influxDB)
	core.Print(core.Stderr(), "  - Inference: %s\n", apiURL)

	if err := app.Run(); err != nil {
		core.Print(core.Stderr(), "%v\n", err)
		core.Exit(1)
	}
}

// spaHandler serves static files with SPA fallback for client-side routing.
func spaHandler(fsys fs.FS) http.Handler {
	fileServer := http.FileServer(http.FS(fsys))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := core.TrimPrefix(r.URL.Path, "/")
		if path == "" {
			path = "index.html"
		}
		if _, err := fs.Stat(fsys, path); err != nil {
			r.URL.Path = "/"
		}
		fileServer.ServeHTTP(w, r)
	})
}

// findDeployDir locates the deploy/ directory relative to the binary.
func findDeployDir() string {
	// Check relative to executable.
	if args := core.Args(); len(args) > 0 && args[0] != "" {
		exe := args[0]
		if abs := core.PathAbs(exe); abs.OK {
			exe = abs.Value.(string)
		}
		dir := core.PathJoin(desktopPathDir(exe), "deploy")
		if core.Stat(core.PathJoin(dir, "docker-compose.yml")).OK {
			return dir
		}
	}
	// Check relative to working directory.
	if cwd := core.Getwd(); cwd.OK {
		dir := core.PathJoin(cwd.Value.(string), "deploy")
		if core.Stat(core.PathJoin(dir, "docker-compose.yml")).OK {
			return dir
		}
	}
	return "deploy"
}

func envOr(key, fallback string) string {
	if v := core.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func desktopPathDir(path string) string {
	sep := byte(core.PathSeparator)
	trimmed := path
	for len(trimmed) > 1 && trimmed[len(trimmed)-1] == sep {
		trimmed = trimmed[:len(trimmed)-1]
	}
	for i := len(trimmed) - 1; i >= 0; i-- {
		if trimmed[i] == sep {
			if i == 0 {
				return string(sep)
			}
			return trimmed[:i]
		}
	}
	return "."
}
