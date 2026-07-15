// SPDX-Licence-Identifier: EUPL-1.2

// Embed the multi-model serve: two model snapshots held resident behind one
// listener, selected per request by alias. serving.LoadModelsConfig reads the
// same declarative --models-config JSON `lem serve -models-config` takes;
// this example writes a minimal 2-model config from two flags instead of
// requiring a hand-authored file (see serving.ModelsConfig for the full shape,
// including memory ceiling and idle-TTL eviction).
//
//	go run ./pkg/serve/multimodel -model1 ~/models/gemma-4-e2b-it-4bit -model2 ~/models/qwen3-8b-it-4bit
//	curl http://127.0.0.1:36914/v1/chat/completions \
//	  -H 'Content-Type: application/json' \
//	  -d '{"model":"small","messages":[{"role":"user","content":"hi"}]}'
//	curl http://127.0.0.1:36914/v1/chat/completions \
//	  -H 'Content-Type: application/json' \
//	  -d '{"model":"big","messages":[{"role":"user","content":"hi"}]}'
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"dappco.re/go/inference/serving"

	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine
)

func main() {
	model1 := flag.String("model1", os.Getenv("LEM_MODEL"), "first model snapshot directory, aliased \"small\"")
	model2 := flag.String("model2", os.Getenv("LEM_MODEL2"), "second model snapshot directory, aliased \"big\"")
	addr := flag.String("addr", "127.0.0.1:36914", "listen address")
	flag.Parse()
	if *model1 == "" || *model2 == "" {
		fmt.Fprintln(os.Stderr, "set -model1 and -model2 to two model snapshot directories")
		os.Exit(2)
	}

	cfgPath, err := writeModelsConfig(*model1, *model2)
	if err != nil {
		fmt.Fprintln(os.Stderr, "config:", err)
		os.Exit(1)
	}
	defer os.Remove(cfgPath)

	// LoadModelsConfig parses the file back into the resolver's spec list plus
	// its residency-budget options — the same call cmd/lem's -models-config
	// flag makes before RunServe.
	specs, opts, err := serving.LoadModelsConfig(cfgPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load config:", err)
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	err = serving.RunServe(ctx, serving.ServeConfig{
		Addr:          *addr,
		Models:        specs,
		MemoryCeiling: opts.MemoryCeiling,
		IdleTTL:       opts.IdleTTL,
		SweepInterval: opts.SweepInterval,
		Log:           os.Stderr,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "serve:", err)
		os.Exit(1)
	}
}

// writeModelsConfig writes a minimal two-model --models-config JSON file to a
// temp path and returns it. Real deployments hand-author this file; the shape
// is serving.ModelsConfig (go/serving/serve_multimodel_config.go) — reused
// directly here rather than mirrored, so the written JSON can never drift
// from what LoadModelsConfig actually parses.
func writeModelsConfig(model1, model2 string) (string, error) {
	cfg := serving.ModelsConfig{
		Models: []serving.ModelSpecConfig{
			{ID: "model1", Path: model1, Aliases: []string{"small"}},
			{ID: "model2", Path: model2, Aliases: []string{"big"}},
		},
	}
	raw, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return "", err
	}
	path := filepath.Join(os.TempDir(), "lem-multimodel-example.json")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		return "", err
	}
	return path, nil
}
