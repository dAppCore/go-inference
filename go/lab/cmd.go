// SPDX-License-Identifier: EUPL-1.2

// Package lab wires the local lab dashboard command into the core CLI.
package lab

import (
	"context"
	"crypto/subtle"
	"net"
	"net/http"
	"os/signal" // Note: retained until lab commands receive a configured core.Signal context.
	"syscall"
	"time"

	"dappco.re/go"
	"dappco.re/go/cli/pkg/cli"
)

const defaultBindAddr = "127.0.0.1:8080"

// CommandOptions configures `core lab serve`.
type CommandOptions struct {
	Bind        string
	AllowRemote bool
}

func init() {
	cli.RegisterCommands(AddLabCommands)
}

// AddLabCommands registers the top-level lab command group.
func AddLabCommands(c *core.Core) core.Result {
	if r := registerLabCommand(c, "lab", core.Command{Description: "Run local lab dashboard and health endpoints."}); !r.OK {
		return r
	}
	return addServeCommand(c, "lab/serve")
}

func registerLabCommand(c *core.Core, path string, command core.Command) core.Result {
	if c.Command(path).OK {
		return core.Ok(nil)
	}
	return c.Command(path, command)
}

func addServeCommand(c *core.Core, path string) core.Result {
	return registerLabCommand(c, path, core.Command{
		Description: "Start the local lab dashboard HTTP server.",
		Flags: core.NewOptions(
			core.Option{Key: "bind", Value: defaultBindAddr},
			core.Option{Key: "allow-remote", Value: false},
		),
		Action: func(opts core.Options) core.Result {
			bind := opts.String("bind")
			if bind == "" {
				bind = defaultBindAddr
			}
			return RunServe(CommandOptions{
				Bind:        bind,
				AllowRemote: opts.Bool("allow-remote"),
			})
		},
	})
}

// notifyServeContext returns a context cancelled on SIGINT/SIGTERM, plus its
// stop function. Package-level so tests can substitute a context they
// control directly (e.g. context.WithCancel) and drive shutdown by calling
// cancel() instead of sending the test process a real OS signal. Default
// behaviour is unchanged — production always gets the real signal.NotifyContext
// wiring below.
var notifyServeContext = func(parent context.Context) (context.Context, context.CancelFunc) {
	return signal.NotifyContext(parent, syscall.SIGINT, syscall.SIGTERM)
}

// RunServe starts the lab dashboard HTTP server.
func RunServe(options CommandOptions) core.Result {
	if r := ValidateBindAddress(options.Bind, options.AllowRemote); !r.OK {
		return r
	}

	authToken := core.Trim(core.Env("CORE_LAB_API_TOKEN"))
	if r := ValidateRemoteAuth(options.AllowRemote, authToken); !r.OK {
		return r
	}

	ctx, stop := notifyServeContext(context.Background())
	defer stop()

	server := &http.Server{
		Addr:         options.Bind,
		Handler:      newServeMux(authToken),
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	errc := make(chan error, 1)
	go func() {
		core.Info("lab dashboard starting", "addr", options.Bind)
		err := server.ListenAndServe()
		if err == http.ErrServerClosed {
			err = nil
		}
		errc <- err
	}()

	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			return core.Fail(err)
		}
		if err := <-errc; err != nil {
			return core.Fail(err)
		}
		return core.Ok(nil)
	case err := <-errc:
		if err != nil {
			return core.Fail(err)
		}
		return core.Ok(nil)
	}
}

func newServeMux(authToken string) *http.ServeMux {
	authWrapper := func(handler http.HandlerFunc) http.HandlerFunc {
		return requireAuth(handler, authToken)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /", authWrapper(index))
	mux.HandleFunc("GET /health", authWrapper(healthz))
	mux.HandleFunc("GET /healthz", authWrapper(healthz))
	return mux
}

// Response bodies are fixed, so they are held as package-level byte slices to
// avoid a per-request []byte conversion on the index/health endpoints (which
// load balancers and liveness probes hit continuously). http.ResponseWriter.Write
// copies the bytes, so the shared slices are never mutated.
var (
	indexBody   = []byte("the inference stack lab\n")
	healthzBody = []byte(`{"status":"ok"}` + "\n")
)

func index(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(indexBody)
}

func healthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(healthzBody)
}

// ValidateBindAddress rejects remote binds unless --allow-remote is set.
func ValidateBindAddress(addr string, allowRemote bool) core.Result {
	if allowRemote || IsLoopbackBindAddress(addr) {
		return core.Ok(nil)
	}
	return core.Fail(core.E("lab.serve", core.Sprintf("refusing to bind lab dashboard to non-loopback address %q without --allow-remote", addr), nil))
}

// IsLoopbackBindAddress reports whether addr binds to a loopback host.
func IsLoopbackBindAddress(addr string) bool {
	host, _, err := net.SplitHostPort(core.Trim(addr))
	if err != nil {
		return false
	}

	if host == "localhost" {
		return true
	}

	ip := net.ParseIP(host)
	if ip == nil {
		return false
	}
	return ip.IsLoopback()
}

func requireAuth(handler http.HandlerFunc, token string) http.HandlerFunc {
	if token == "" {
		return handler
	}

	// The expected header is fixed once the token is known, so build it (and the
	// byte form the constant-time compare needs) at wrap time rather than on
	// every request.
	expected := core.Concat("Bearer ", token)
	expectedBytes := []byte(expected)

	return func(w http.ResponseWriter, r *http.Request) {
		authHeader := core.Trim(r.Header.Get("Authorization"))
		if len(authHeader) != len(expected) || subtle.ConstantTimeCompare([]byte(authHeader), expectedBytes) != 1 {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}

		handler(w, r)
	}
}

// ValidateRemoteAuth requires CORE_LAB_API_TOKEN before remote access is enabled.
func ValidateRemoteAuth(allowRemote bool, authToken string) core.Result {
	if !allowRemote || core.Trim(authToken) != "" {
		return core.Ok(nil)
	}
	return core.Fail(core.E("lab.serve", "refusing to start lab dashboard with --allow-remote without CORE_LAB_API_TOKEN", nil))
}
