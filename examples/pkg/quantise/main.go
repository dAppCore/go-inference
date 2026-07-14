// SPDX-Licence-Identifier: EUPL-1.2

// Quantise a dense bf16/f32 safetensors model pack into GGUF — the same call
// `lem quant -gguf <format>` wraps (go/cmd/lem/quant.go). GGUF's quantisation
// schemes are named recipes (q4_k_m, q8_0, ...), not a raw bit count, so the
// -bits flag here is a small convenience mapping onto the nearest recipe;
// cmd/lem's own -gguf flag takes the recipe name directly for full control.
//
//	go run ./pkg/quantise -src ~/models/gemma-4-e2b-it-bf16 -out ~/models/gemma-4-e2b-it-gguf-q4
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/gguf"
)

func main() {
	src := flag.String("src", "", "dense bf16/f32 safetensors model directory")
	out := flag.String("out", "", "output GGUF model directory")
	bits := flag.Int("bits", 4, "quantisation width (2, 3, 4, 5, 6, or 8) — mapped onto the nearest GGUF recipe")
	flag.Parse()
	if *src == "" || *out == "" {
		fmt.Fprintln(os.Stderr, "set -src and -out")
		os.Exit(2)
	}

	weightFiles, globErr := filepath.Glob(filepath.Join(*src, "*.safetensors"))
	if globErr != nil || len(weightFiles) == 0 {
		fmt.Fprintln(os.Stderr, "no *.safetensors shards under", *src)
		os.Exit(1)
	}
	sort.Strings(weightFiles)
	arch, _, _ := model.ProbeDirArch(*src) // best-effort architecture for GGUF metadata

	res, err := gguf.QuantizeModelPack(context.Background(), gguf.QuantizeOptions{
		SourcePack: gguf.Source{Root: *src, Architecture: arch, WeightFiles: weightFiles},
		OutputPath: *out,
		Format:     ggufFormatForBits(*bits),
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "quantise:", err)
		os.Exit(1)
	}

	fmt.Printf("wrote %s\n", res.WeightPath)
	fmt.Printf("format: requested %s, used %s\n", res.RequestedFormat, res.Format)
	fmt.Printf("tensors: %d quantised of %d\n", res.QuantizedTensors, res.TensorCount)
	for _, note := range res.Notes {
		fmt.Println("note:", note)
	}
}

// ggufFormatForBits maps a bit-width onto the closest named GGUF recipe.
// GGUF has no raw-bits knob (see gguf.QuantizeOptions.Format); cmd/lem's own
// -gguf flag takes the recipe name directly (q4_k_m, q8_0, ...).
func ggufFormatForBits(bits int) gguf.QuantizeFormat {
	switch {
	case bits <= 2:
		return gguf.QuantizeQ2_K_M
	case bits == 3:
		return gguf.QuantizeQ3_K_M
	case bits == 5:
		return gguf.QuantizeQ5_K_M
	case bits == 6:
		return gguf.QuantizeQ6_K
	case bits >= 8:
		return gguf.QuantizeQ8_0
	default:
		return gguf.QuantizeQ4_K_M
	}
}
