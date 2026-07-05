package main

import (
	"context"
	"sync"
	"time"

	core "dappco.re/go"
	"github.com/wailsapp/wails/v3/pkg/application"
	execabs "golang.org/x/sys/execabs"
)

// DockerService manages the LEM Docker compose stack.
// Provides start/stop/status for Forgejo, InfluxDB, and inference services.
type DockerService struct {
	composeFile string
	mu          sync.RWMutex
	services    map[string]ContainerStatus
}

// ContainerStatus represents a Docker container's state.
type ContainerStatus struct {
	Name    string `json:"name"`
	Image   string `json:"image"`
	Status  string `json:"status"`
	Health  string `json:"health"`
	Ports   string `json:"ports"`
	Running bool   `json:"running"`
}

// StackStatus represents the overall stack state.
type StackStatus struct {
	Running    bool                       `json:"running"`
	Services   map[string]ContainerStatus `json:"services"`
	ComposeDir string                     `json:"composeDir"`
}

// NewDockerService creates a DockerService.
// composeDir should point to the deploy/ directory containing docker-compose.yml.
func NewDockerService(composeDir string) *DockerService {
	return &DockerService{
		composeFile: core.PathJoin(composeDir, "docker-compose.yml"),
		services:    make(map[string]ContainerStatus),
	}
}

// ServiceName returns the Wails service name.
func (d *DockerService) ServiceName() string {
	return "DockerService"
}

// ServiceStartup is called when the Wails app starts.
func (d *DockerService) ServiceStartup(ctx context.Context, options application.ServiceOptions) core.Result {
	core.Print(core.Stderr(), "DockerService started\n")
	go d.statusLoop(ctx)
	return core.Ok(nil)
}

// Start brings up the full Docker compose stack.
func (d *DockerService) Start() core.Result {
	core.Print(core.Stderr(), "Starting LEM stack...\n")
	return d.compose("up", "-d")
}

// Stop takes down the Docker compose stack.
func (d *DockerService) Stop() core.Result {
	core.Print(core.Stderr(), "Stopping LEM stack...\n")
	return d.compose("down")
}

// Restart restarts the full stack.
func (d *DockerService) Restart() core.Result {
	if r := d.Stop(); !r.OK {
		return r
	}
	return d.Start()
}

// StartService starts a single service.
func (d *DockerService) StartService(name string) core.Result {
	return d.compose("up", "-d", name)
}

// StopService stops a single service.
func (d *DockerService) StopService(name string) core.Result {
	return d.compose("stop", name)
}

// RestartService restarts a single service.
func (d *DockerService) RestartService(name string) core.Result {
	return d.compose("restart", name)
}

// Logs returns recent logs for a service.
func (d *DockerService) Logs(name string, lines int) core.Result {
	if lines <= 0 {
		lines = 50
	}
	outResult := d.composeOutput("logs", "--tail", core.Sprintf("%d", lines), "--no-color", name)
	if !outResult.OK {
		return outResult
	}
	return outResult
}

// GetStatus returns the current stack status.
func (d *DockerService) GetStatus() StackStatus {
	d.mu.RLock()
	defer d.mu.RUnlock()

	running := false
	for _, s := range d.services {
		if s.Running {
			running = true
			break
		}
	}

	return StackStatus{
		Running:    running,
		Services:   d.services,
		ComposeDir: desktopPathDir(d.composeFile),
	}
}

// IsRunning returns whether any services are running.
func (d *DockerService) IsRunning() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	for _, s := range d.services {
		if s.Running {
			return true
		}
	}
	return false
}

// Pull pulls latest images for all services.
func (d *DockerService) Pull() core.Result {
	return d.compose("pull")
}

func (d *DockerService) compose(args ...string) core.Result {
	fullArgs := append([]string{"compose", "-f", d.composeFile}, args...)
	cmd := execabs.Command("docker", fullArgs...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return core.Fail(core.Errorf("docker compose %s: %w: %s", core.Join(" ", args...), err, string(out)))
	}
	return core.Ok(nil)
}

func (d *DockerService) composeOutput(args ...string) core.Result {
	fullArgs := append([]string{"compose", "-f", d.composeFile}, args...)
	cmd := execabs.Command("docker", fullArgs...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return core.Fail(core.Errorf("docker compose %s: %w: %s", core.Join(" ", args...), err, string(out)))
	}
	return core.Ok(string(out))
}

func (d *DockerService) refreshStatus() {
	outResult := d.composeOutput("ps", "--format", "json")
	if !outResult.OK {
		return
	}
	out := outResult.Value.(string)

	d.mu.Lock()
	defer d.mu.Unlock()

	d.services = make(map[string]ContainerStatus)

	// docker compose ps --format json outputs one JSON object per line.
	for _, line := range core.Split(core.Trim(out), "\n") {
		if line == "" {
			continue
		}
		var container struct {
			Name    string `json:"Name"`
			Image   string `json:"Image"`
			Service string `json:"Service"`
			Status  string `json:"Status"`
			Health  string `json:"Health"`
			State   string `json:"State"`
			Ports   string `json:"Ports"`
		}
		if r := core.JSONUnmarshal([]byte(line), &container); !r.OK {
			continue
		}

		name := container.Service
		if name == "" {
			name = container.Name
		}

		d.services[name] = ContainerStatus{
			Name:    container.Name,
			Image:   container.Image,
			Status:  container.Status,
			Health:  container.Health,
			Ports:   container.Ports,
			Running: container.State == "running",
		}
	}
}

func (d *DockerService) statusLoop(ctx context.Context) {
	d.refreshStatus()

	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			d.refreshStatus()
		}
	}
}
