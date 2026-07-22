// SPDX-License-Identifier: EUPL-1.2

// Package gitserver embeds a private loopback Git control plane for agent work.
package gitserver

import (
	"context"
	"time"

	core "dappco.re/go"
	processapi "dappco.re/go/process/pkg/api"
)

// Options configures the private Soft Serve data and listener boundary.
type Options struct {
	DataPath        string
	ListenAddress   string
	PublicURL       string
	ShutdownTimeout time.Duration
	PID             func() int
	ProcessAlive    func(int) bool
}

// Repository is the credential-free clone location and separate SSH trust material.
type Repository struct {
	Name           string
	CloneURL       string
	IdentityFile   string
	KnownHostsFile string
}

// Health is a point-in-time view of the private Git listener.
type Health struct {
	Running bool
	Address string
	Reason  string
}

// Service is the narrow Git control-plane contract used by workspace management.
type Service interface {
	Start(context.Context) core.Result
	EnsureRepository(context.Context, string) core.Result
	Health(context.Context) core.Result
	Close() core.Result
}

// DefaultOptions returns a restrictive loopback-only Soft Serve configuration.
func DefaultOptions(dataPath string) core.Result {
	dataPath = core.Trim(dataPath)
	if dataPath == "" {
		return core.Fail(core.NewError("agent git server data path is required"))
	}
	if !core.PathIsAbs(dataPath) {
		return core.Fail(core.NewError("agent git server data path must be absolute"))
	}
	return core.Ok(Options{
		DataPath:        dataPath,
		ListenAddress:   "127.0.0.1:23231",
		PublicURL:       "ssh://127.0.0.1:23231",
		ShutdownTimeout: 5 * time.Second,
		PID:             core.Getpid,
		ProcessAlive:    processapi.PIDAlive,
	})
}
