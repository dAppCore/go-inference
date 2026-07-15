// SPDX-Licence-Identifier: EUPL-1.2

// Embed the OpenAI-compatible server in your own Go app: one call to
// serving.RunServe hosts the same OpenAI/Anthropic/Ollama-compatible mux
// `lem serve` does, but wired into a process you control (no cmd/lem
// subprocess). RunServe blocks until ctx is cancelled, so Ctrl-C here shuts
// the listener down cleanly.
//
//	go run ./pkg/serve -model ~/models/gemma-4-e2b-it-4bit
//	curl http://127.0.0.1:36912/v1/chat/completions \
//	  -H 'Content-Type: application/json' \
//	  -d '{"model":"gemma-4-e2b-it-4bit","messages":[{"role":"user","content":"hi"}]}'
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"dappco.re/go/inference/serving"

	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "model snapshot directory (config.json + *.safetensors)")
	addr := flag.String("addr", "127.0.0.1:36912", "listen address")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a model snapshot directory")
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// ServeConfig is the declarative serve request; RunServe resolves the
	// drafter/profile/continuity wiring from it. Everything left at its zero
	// value degrades cleanly: no admin token (the /v1/admin/* subtree stays
	// open), no speculative loader (plain autoregressive), no continuity
	// enabler (stateless chat) — see serving.ServeConfig's field comments for
	// what each zero value means before adding one back.
	err := serving.RunServe(ctx, serving.ServeConfig{
		Addr:      *addr,
		ModelPath: *model,
		Log:       os.Stderr,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "serve:", err)
		os.Exit(1)
	}
}
