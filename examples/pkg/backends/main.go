// SPDX-Licence-Identifier: EUPL-1.2

// The backend registry: every engine that blank-imports itself registers with
// inference.Register at init() time, and inference.List/All/Get/Default read
// that registry back. This example blank-imports both engines this repo
// ships — the platform engine (metal on darwin/arm64, via
// examples/internal/engine) and engine/hip (registers "rocm"; off
// linux/amd64 it registers the honest stub with Available() false rather
// than not registering at all) — so the printed list is platform-truthful:
// what you see is what would actually load on THIS machine, not a hardcoded
// menu.
//
//	go run ./pkg/backends                              # zero-hardware: print the registry, exit
//	go run ./pkg/backends -model ~/models/gemma-4-e2b-it-4bit  # also load, pinned to the default backend
package main

import (
	"flag"
	"fmt"
	"os"

	"dappco.re/go/inference"
	_ "dappco.re/go/inference/engine/hip" // registers "rocm" (real on linux/amd64, an honest unavailable stub elsewhere)

	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine (e.g. "metal" on darwin/arm64)
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "model snapshot directory; unset just prints the registry and exits")
	flag.Parse()

	// List returns every registered name, sorted — the quick "what's compiled
	// in" check.
	fmt.Println("registered backends:", inference.List())

	// All ranges (name, Backend) pairs so a caller can also read per-backend
	// state such as Available() without a second lookup per name.
	for name, b := range inference.All() {
		fmt.Printf("  %-10s available=%v\n", name, b.Available())
	}

	// Get looks up one backend by name directly, without scanning the registry.
	if b, ok := inference.Get("metal"); ok {
		fmt.Println("metal is registered; available:", b.Available())
	}

	if *model == "" {
		return // the zero-hardware path: nothing to load, registry already printed
	}

	// Default walks the preference order (metal -> rocm -> llama_cpp -> any)
	// and picks the first Available() one.
	def := inference.Default()
	if !def.OK {
		fmt.Fprintln(os.Stderr, "default:", def.Value)
		os.Exit(1)
	}
	backend := def.Value.(inference.Backend)

	// WithBackend pins the load to a specific name instead of re-running the
	// preference scan — here pinned to whatever Default() just resolved, so
	// the two agree on this machine.
	r := inference.LoadModel(*model, inference.WithBackend(backend.Name()))
	if !r.OK {
		fmt.Fprintln(os.Stderr, "load:", r.Value)
		os.Exit(1)
	}
	m := r.Value.(inference.TextModel)
	defer m.Close()
	fmt.Println("loaded via backend:", backend.Name())
}
