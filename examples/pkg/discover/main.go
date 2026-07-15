// SPDX-Licence-Identifier: EUPL-1.2

// Discover walks a directory tree looking for model snapshots — any directory
// containing config.json plus at least one *.safetensors file. It needs no
// loaded model and no GPU engine, so unlike the other examples in this tree
// it does not blank-import examples/internal/engine.
//
//	go run ./pkg/discover -dir ~/.cache/huggingface/hub
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"dappco.re/go/inference"
)

func main() {
	home, _ := os.UserHomeDir()
	defaultDir := filepath.Join(home, ".cache", "huggingface", "hub")
	dir := flag.String("dir", defaultDir, "directory to scan for model snapshots")
	flag.Parse()

	found := 0
	for model := range inference.Discover(*dir) {
		found++
		fmt.Printf("%s  (%s)\n", model.Path, model.ModelType)
	}
	if found == 0 {
		fmt.Printf("no model snapshots found under %s\n", *dir)
	}
}
