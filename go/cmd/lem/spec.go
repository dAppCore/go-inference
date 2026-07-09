// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"

	core "dappco.re/go"
	goapi "dappco.re/go/api"
	"dappco.re/go/inference/engine/driver"
	"dappco.re/go/inference/kv/sessionkv"
	infapi "dappco.re/go/inference/serving/api"
)

// runSpecCommand builds the OpenAPI document for lem's HTTP surface from the
// core/api RouteGroups (their Describe() methods) and writes it to a file — the
// machine-readable API definition the SDK generators consume (see the `sdk`
// Taskfile target). Thin: construct the describable groups, export.
//
// No model is loaded and no server is started. Every group's Describe() returns
// static route metadata, so the constructors take nil services here — the spec
// is a compile-time property of the API surface, not a runtime one.
func runSpecCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("spec"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	output := fs.String("o", "build/sdk/openapi.json", "output path for the OpenAPI document")
	format := fs.String("format", "json", "spec format: json or yaml")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	engine, err := goapi.New(goapi.WithSwagger(
		"Lethean lem API",
		"The HTTP surface of the sovereign lem inference binary: non-LLM scoring, behavioural embeddings, driver orchestration, and OpenAI/Anthropic/Ollama-compatible inference.",
		"0.1.0",
	))
	if err != nil {
		core.Print(stderr, "%s: %v\n", cliCommandName("spec"), err)
		return 1
	}
	// The describable route groups that make up lem's HTTP surface. nil services
	// are fine — Describe() is service-free (static route metadata).
	engine.Register(infapi.New())                     // /v1        score + behavioural-embedding
	engine.Register(infapi.NewRoutes(nil))            // /v1/ml     backends, status, generate
	engine.Register(driver.NewProvider(nil))          // /v1/driver models, serve, status, stop
	engine.Register(driver.NewInferenceProvider(nil)) // /v1        chat/completions, messages, models
	// session.kv (/v1/state) needs a store to construct; a throwaway temp store
	// is enough to surface its static Describe(). Best-effort — a store-open
	// failure omits the two /v1/state read routes rather than failing the export.
	if host, herr := sessionkv.Open(ctx, core.PathJoin(core.TempDir(), "lem-spec-session.kv")); herr == nil {
		defer host.Close()
		engine.Register(host)
	}

	if err := goapi.ExportSpecToFile(*output, *format, engine.OpenAPISpecBuilder(), engine.Groups()); err != nil {
		core.Print(stderr, "%s: write spec: %v\n", cliCommandName("spec"), err)
		return 1
	}
	core.Print(stdout, "OpenAPI spec written to %s (%d route groups)\n", *output, len(engine.Groups()))
	return 0
}
