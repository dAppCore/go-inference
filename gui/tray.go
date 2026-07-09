package main

import (
	"context"
	"runtime"

	core "dappco.re/go"
	"github.com/wailsapp/wails/v3/pkg/application"
	execabs "golang.org/x/sys/execabs"
)

// TrayService provides system tray bindings for the LEM desktop.
// Exposes status to the frontend and controls the tray menu.
type TrayService struct {
	app       *application.App
	dashboard *DashboardService
	docker    *DockerService
	container *ContainerService
	agent     *AgentRunner
}

// NewTrayService creates a new TrayService.
func NewTrayService(app *application.App) *TrayService {
	return &TrayService{app: app}
}

// SetServices wires up service references after app creation.
func (t *TrayService) SetServices(dashboard *DashboardService, docker *DockerService, contained *ContainerService, agent *AgentRunner) {
	t.dashboard = dashboard
	t.docker = docker
	t.container = contained
	t.agent = agent
}

// ServiceName returns the Wails service name.
func (t *TrayService) ServiceName() string {
	return "TrayService"
}

// ServiceStartup is called when the Wails app starts.
func (t *TrayService) ServiceStartup(ctx context.Context, options application.ServiceOptions) core.Result {
	core.Print(core.Stderr(), "TrayService started\n")
	return core.Ok(nil)
}

// ServiceShutdown is called on app exit.
func (t *TrayService) ServiceShutdown() core.Result {
	core.Print(core.Stderr(), "TrayService shutdown\n")
	return core.Ok(nil)
}

// TraySnapshot is the complete tray state for the frontend.
type TraySnapshot struct {
	StackRunning     bool            `json:"stackRunning"`
	ContainedRunning bool            `json:"containedRunning"`
	ContainedStatus  ContainerStatus `json:"containedStatus"`
	AgentRunning     bool            `json:"agentRunning"`
	AgentTask        string          `json:"agentTask"`
	Training         []TrainingRow   `json:"training"`
	Generation       GenerationStats `json:"generation"`
	Models           []ModelInfo     `json:"models"`
	DockerServices   int             `json:"dockerServices"`
}

// GetSnapshot returns the full tray state.
func (t *TrayService) GetSnapshot() TraySnapshot {
	snap := TraySnapshot{}

	if t.dashboard != nil {
		ds := t.dashboard.GetSnapshot()
		snap.Training = ds.Training
		snap.Generation = ds.Generation
		snap.Models = ds.Models
	}

	if t.docker != nil {
		status := t.docker.GetStatus()
		snap.StackRunning = status.Running
		snap.DockerServices = len(status.Services)
	}

	if t.container != nil {
		snap.ContainedStatus = t.container.GetStatus()
		snap.ContainedRunning = snap.ContainedStatus.Running
	}

	if t.agent != nil {
		snap.AgentRunning = t.agent.IsRunning()
		snap.AgentTask = t.agent.CurrentTask()
	}

	return snap
}

// StartStack starts the Docker compose stack.
func (t *TrayService) StartStack() core.Result {
	if t.docker == nil {
		return core.Fail(core.E("lem.desktop.tray", "docker service not available", nil))
	}
	return t.docker.Start()
}

// StopStack stops the Docker compose stack.
func (t *TrayService) StopStack() core.Result {
	if t.docker == nil {
		return core.Fail(core.E("lem.desktop.tray", "docker service not available", nil))
	}
	return t.docker.Stop()
}

// StartContained launches the Apple container service.
func (t *TrayService) StartContained() core.Result {
	if t.container == nil {
		return core.Fail(core.E("lem.desktop.tray", "container service not available", nil))
	}
	return t.container.Start()
}

// StopContained stops the Apple container service.
func (t *TrayService) StopContained() core.Result {
	if t.container == nil {
		return core.Fail(core.E("lem.desktop.tray", "container service not available", nil))
	}
	return t.container.Stop()
}

// StartAgent starts the scoring agent.
func (t *TrayService) StartAgent() core.Result {
	if t.agent == nil {
		return core.Fail(core.E("lem.desktop.tray", "agent service not available", nil))
	}
	return t.agent.Start()
}

// StopAgent stops the scoring agent.
func (t *TrayService) StopAgent() {
	if t.agent != nil {
		t.agent.Stop()
	}
}

// setupSystemTray configures the system tray icon and menu.
func setupSystemTray(app *application.App, tray *TrayService, dashboard *DashboardService, docker *DockerService, contained *ContainerService) {
	systray := app.SystemTray.New()
	systray.SetTooltip("LEM — Lethean Ethics Model")

	// Platform-specific icon.
	if runtime.GOOS == "darwin" {
		systray.SetTemplateIcon(trayIconTemplate)
	} else {
		systray.SetDarkModeIcon(trayIconDark)
		systray.SetIcon(trayIconLight)
	}

	// ── Tray Panel (frameless dropdown) ──
	trayWindow := app.Window.NewWithOptions(application.WebviewWindowOptions{
		Name:             "tray-panel",
		Title:            "LEM",
		Width:            420,
		Height:           520,
		URL:              "/tray",
		Hidden:           true,
		Frameless:        true,
		BackgroundColour: application.NewRGB(15, 23, 42),
	})
	systray.AttachWindow(trayWindow).WindowOffset(5)

	// ── Dashboard Window ──
	app.Window.NewWithOptions(application.WebviewWindowOptions{
		Name:             "dashboard",
		Title:            "LEM Dashboard",
		Width:            1400,
		Height:           900,
		URL:              "/dashboard",
		Hidden:           true,
		BackgroundColour: application.NewRGB(15, 23, 42),
	})

	// ── Workbench Window (model scoring, probes) ──
	app.Window.NewWithOptions(application.WebviewWindowOptions{
		Name:             "workbench",
		Title:            "LEM Workbench",
		Width:            1200,
		Height:           800,
		URL:              "/workbench",
		Hidden:           true,
		BackgroundColour: application.NewRGB(15, 23, 42),
	})

	// ── Settings Window ──
	app.Window.NewWithOptions(application.WebviewWindowOptions{
		Name:             "settings",
		Title:            "LEM Settings",
		Width:            600,
		Height:           500,
		URL:              "/settings",
		Hidden:           true,
		BackgroundColour: application.NewRGB(15, 23, 42),
	})

	// ── Build Tray Menu ──
	trayMenu := app.Menu.New()

	// Status (dynamic).
	statusItem := trayMenu.Add("LEM: Idle")
	statusItem.SetEnabled(false)

	trayMenu.AddSeparator()

	// Stack control.
	stackItem := trayMenu.Add("Start Services")
	stackItem.OnClick(func(ctx *application.Context) {
		if docker.IsRunning() {
			docker.Stop()
			stackItem.SetLabel("Start Services")
			statusItem.SetLabel("LEM: Stopped")
		} else {
			docker.Start()
			stackItem.SetLabel("Stop Services")
			statusItem.SetLabel("LEM: Running")
		}
	})

	// Contained service control (Apple containers via go-container).
	containedItem := trayMenu.Add("Start Contained Service")
	containedItem.OnClick(func(ctx *application.Context) {
		if contained == nil {
			return
		}
		if contained.IsRunning() {
			contained.Stop()
			containedItem.SetLabel("Start Contained Service")
		} else {
			contained.Start()
			containedItem.SetLabel("Stop Contained Service")
		}
	})

	// Agent control.
	agentItem := trayMenu.Add("Start Scoring Agent")
	agentItem.OnClick(func(ctx *application.Context) {
		if tray.agent != nil && tray.agent.IsRunning() {
			tray.agent.Stop()
			agentItem.SetLabel("Start Scoring Agent")
		} else if tray.agent != nil {
			tray.agent.Start()
			agentItem.SetLabel("Stop Scoring Agent")
		}
	})

	trayMenu.AddSeparator()

	// Windows.
	trayMenu.Add("Open Dashboard").OnClick(func(ctx *application.Context) {
		if w, ok := app.Window.Get("dashboard"); ok {
			w.Show()
			w.Focus()
		}
	})

	trayMenu.Add("Open Workbench").OnClick(func(ctx *application.Context) {
		if w, ok := app.Window.Get("workbench"); ok {
			w.Show()
			w.Focus()
		}
	})

	trayMenu.Add("Open Forge").OnClick(func(ctx *application.Context) {
		// Open the local Forgejo in the default browser.
		openBrowser("http://localhost:3000")
	})

	trayMenu.AddSeparator()

	// Stats submenu.
	statsMenu := trayMenu.AddSubmenu("Training")
	statsMenu.Add("Golden Set: loading...").SetEnabled(false)
	statsMenu.Add("Expansion: loading...").SetEnabled(false)
	statsMenu.Add("Models Scored: loading...").SetEnabled(false)

	trayMenu.AddSeparator()

	// Settings.
	trayMenu.Add("Settings...").OnClick(func(ctx *application.Context) {
		if w, ok := app.Window.Get("settings"); ok {
			w.Show()
			w.Focus()
		}
	})

	trayMenu.AddSeparator()

	// Quit.
	trayMenu.Add("Quit LEM").OnClick(func(ctx *application.Context) {
		app.Quit()
	})

	systray.SetMenu(trayMenu)
}

// openBrowser launches the default browser.
func openBrowser(url string) {
	var cmd string
	var args []string
	switch runtime.GOOS {
	case "darwin":
		cmd = "open"
	case "linux":
		cmd = "xdg-open"
	case "windows":
		cmd = "rundll32"
		args = []string{"url.dll,FileProtocolHandler"}
	}
	if cmd == "" {
		return
	}
	args = append(args, url)
	go func() {
		if err := execabs.Command(cmd, args...).Start(); err != nil {
			core.Print(core.Stderr(), "open browser: %v\n", err)
		}
	}()
}
