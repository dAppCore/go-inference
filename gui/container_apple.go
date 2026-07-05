package main

import (
	"context"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/container"
	"github.com/wailsapp/wails/v3/pkg/application"
)

// defaultContainedImage is the OCI image launched by ContainerService.Start
// when no explicit image is configured. A tiny base image keeps the first
// App Store slice fast to pull and boot.
const defaultContainedImage = "docker.io/library/alpine:latest"

// ContainerService manages a single Apple container for the LEM Runtime.
//
// It mirrors DockerService's shape — a Wails service exposing
// Start/Stop/GetStatus/IsRunning plus a background statusLoop — but is backed
// by go-container's Apple provider (the `container` CLI shipped with macOS 26+)
// rather than `docker compose`. This is the first slice toward shipping the LEM
// Runtime GUI on the App Store via Apple Containerisation; it runs ALONGSIDE
// DockerService rather than replacing it.
//
// Usage:
//
//	svc := NewContainerService("my-service", "docker.io/library/alpine:latest")
//	svc.Start()
//	status := svc.GetStatus()
type ContainerService struct {
	provider *container.AppleProvider
	name     string
	image    string

	mu        sync.RWMutex
	container ContainerStatus
}

// NewContainerService creates a ContainerService.
// name is the Apple container name (also its id); image is the OCI reference to
// launch. An empty image falls back to defaultContainedImage so the slice has a
// sensible default.
func NewContainerService(name, image string) *ContainerService {
	if name == "" {
		name = "lem-contained"
	}
	if image == "" {
		image = defaultContainedImage
	}
	return &ContainerService{
		provider: container.NewAppleProvider(),
		name:     name,
		image:    image,
	}
}

// ServiceName returns the Wails service name.
func (c *ContainerService) ServiceName() string {
	return "ContainerService"
}

// ServiceStartup is called when the Wails app starts.
func (c *ContainerService) ServiceStartup(ctx context.Context, options application.ServiceOptions) core.Result {
	core.Print(core.Stderr(), "ContainerService started\n")
	go c.statusLoop(ctx)
	return core.Ok(nil)
}

// Available reports whether the Apple container runtime can run on this host.
func (c *ContainerService) Available() bool {
	if c == nil || c.provider == nil {
		return false
	}
	return c.provider.Available()
}

// Start launches the configured Apple container in the background.
// It returns the resulting *container.Container on success.
func (c *ContainerService) Start() core.Result {
	core.Print(core.Stderr(), "Starting contained service %s (%s)...\n", c.name, c.image)
	if c.provider == nil {
		return core.Fail(core.E("lem.desktop.container", "apple container provider not available", nil))
	}

	img := &container.Image{
		Name:     c.image,
		Path:     c.image,
		Format:   container.FormatOCI,
		Provider: string(container.RuntimeApple),
	}

	runResult := c.provider.Run(img,
		container.WithName(c.name),
		container.WithDetach(true),
	)
	if !runResult.OK {
		return runResult
	}

	ctr := core.MustCast[*container.Container](runResult)
	c.applyContainer(ctr)
	return runResult
}

// Stop stops the running Apple container by name.
func (c *ContainerService) Stop() core.Result {
	core.Print(core.Stderr(), "Stopping contained service %s...\n", c.name)
	if c.provider == nil {
		return core.Fail(core.E("lem.desktop.container", "apple container provider not available", nil))
	}
	r := c.provider.Stop(c.name)
	if r.OK {
		c.refreshStatus()
	}
	return r
}

// Restart stops then starts the contained service.
func (c *ContainerService) Restart() core.Result {
	if r := c.Stop(); !r.OK {
		return r
	}
	return c.Start()
}

// Logs returns recent logs for the contained service.
func (c *ContainerService) Logs(lines int) core.Result {
	if c.provider == nil {
		return core.Fail(core.E("lem.desktop.container", "apple container provider not available", nil))
	}
	if lines <= 0 {
		lines = 50
	}
	return c.provider.Logs(c.name, lines)
}

// GetStatus returns the current contained-service status.
func (c *ContainerService) GetStatus() ContainerStatus {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.container
}

// IsRunning reports whether the contained service is currently running.
func (c *ContainerService) IsRunning() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.container.Running
}

// applyContainer records a launched container's state into the status field.
func (c *ContainerService) applyContainer(ctr *container.Container) {
	if ctr == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.container = containerStatusFrom(ctr)
}

// refreshStatus re-reads the contained service's state from the provider and
// records it. The provider tracks containers it launched, so a launched-then-
// exited container reflects its final state here.
func (c *ContainerService) refreshStatus() {
	if c.provider == nil {
		return
	}

	// Prefer the provider's tracked set — it carries the live status of
	// containers this process launched without shelling the CLI again.
	for _, ctr := range c.provider.Tracked() {
		if ctr.ID == c.name || ctr.Name == c.name {
			c.applyContainer(ctr)
			return
		}
	}

	// Fall back to a CLI inspect for containers started in a prior session.
	inspectResult := c.provider.Inspect(c.name)
	if !inspectResult.OK {
		return
	}
	c.applyContainer(core.MustCast[*container.Container](inspectResult))
}

func (c *ContainerService) statusLoop(ctx context.Context) {
	c.refreshStatus()

	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.refreshStatus()
		}
	}
}

// containerStatusFrom maps a go-container Container into the ContainerStatus
// shape the frontend already consumes for Docker services.
func containerStatusFrom(ctr *container.Container) ContainerStatus {
	ports := ""
	for host, guest := range ctr.Ports {
		if ports != "" {
			ports += ", "
		}
		ports += core.Sprintf("%d:%d", host, guest)
	}
	return ContainerStatus{
		Name:    ctr.Name,
		Image:   ctr.Image,
		Status:  string(ctr.Status),
		Ports:   ports,
		Running: ctr.Status == container.StatusRunning,
	}
}
