// SPDX-Licence-Identifier: EUPL-1.2

// Quantise a dense bf16/f32 safetensors model pack into any of lem's output
// formats — the same calls `lem quant` wraps (cli/quant.go). Two families:
//
//   - gguf (default): the portable interchange lane for the llama.cpp
//     ecosystem. GGUF's quantisation schemes are named recipes (q4_k_m,
//     q8_0, ...), not a raw bit count, so -bits maps onto the nearest recipe;
//     `lem quant --gguf` takes the recipe name directly for full control.
//   - gptq | awq | fp8 | nf4: the HF-ecosystem exporters — GPTQ and AutoAWQ
//     GEMM packing for vLLM/TGI/ExLlama-class consumers, compressed-tensors
//     static E4M3, and bitsandbytes NF4 blockwise. GPTQ/AWQ honour -bits and
//     -group; fp8 and NF4 carry their formats' own fixed shapes.
//
// The engine's own native format (MLX group-affine, both engines load it
// directly) is `lem quant`'s default lane and lives in model/quant/mlxaffine.
//
//	go run ./pkg/quantise -src ~/models/gemma-4-e2b-it-bf16 -out /tmp/e2b-q4
//	go run ./pkg/quantise -format gptq -src ~/models/gemma-4-e2b-it-bf16 -out /tmp/e2b-gptq
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
	"dappco.re/go/inference/model/quant/awq"
	"dappco.re/go/inference/model/quant/fp8"
	"dappco.re/go/inference/model/quant/gptq"
	"dappco.re/go/inference/model/quant/nf4"
)

func main() {
	src := flag.String("src", "", "dense bf16/f32 safetensors model directory")
	out := flag.String("out", "", "output model directory (or .gguf path for -format gguf)")
	format := flag.String("format", "gguf", "output format: gguf, gptq, awq, fp8, or nf4")
	bits := flag.Int("bits", 4, "quantisation width — gguf maps onto the nearest recipe; gptq/awq use it directly")
	group := flag.Int("group", 128, "group size for the gptq/awq lanes")
	flag.Parse()
	if *src == "" || *out == "" {
		fmt.Fprintln(os.Stderr, "set -src and -out")
		os.Exit(2)
	}

	ctx := context.Background()
	progress := func(name string, quantised bool, done, total int) {
		if done == total || done%50 == 0 {
			fmt.Printf("\r%d/%d %s", done, total, name)
		}
		if done == total {
			fmt.Println()
		}
		_ = quantised
	}

	switch *format {
	case "gguf":
		weightFiles, globErr := filepath.Glob(filepath.Join(*src, "*.safetensors"))
		if globErr != nil || len(weightFiles) == 0 {
			fmt.Fprintln(os.Stderr, "no *.safetensors shards under", *src)
			os.Exit(1)
		}
		sort.Strings(weightFiles)
		arch, _, _ := model.ProbeDirArch(*src) // best-effort architecture for GGUF metadata
		res, err := gguf.QuantizeModelPack(ctx, gguf.QuantizeOptions{
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
	case "gptq":
		res, err := gptq.ConvertSnapshot(ctx, *src, *out, gptq.Options{Bits: *bits, GroupSize: *group, Symmetric: true}, progress)
		summarise("gptq", res, err)
	case "awq":
		res, err := awq.ConvertSnapshot(ctx, *src, *out, awq.Options{Bits: *bits, GroupSize: *group, ZeroPoint: true}, progress)
		summarise("awq", (*hfResult)(res), err)
	case "fp8":
		res, err := fp8.ConvertSnapshot(ctx, *src, *out, progress)
		summarise("fp8", (*hfResult)(res), err)
	case "nf4":
		res, err := nf4.ConvertSnapshot(ctx, *src, *out, progress)
		summarise("nf4", (*hfResult)(res), err)
	default:
		fmt.Fprintln(os.Stderr, "unknown -format (want gguf, gptq, awq, fp8, or nf4):", *format)
		os.Exit(2)
	}
}

// hfResult is the shared shape every HF exporter's Result carries; the named
// conversions above only re-brand identical structs so one summary fits all.
type hfResult = gptq.Result

func summarise(lane string, res *hfResult, err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, lane+":", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s (%s + %s)\n", res.OutputDir, filepath.Base(res.WeightFile), filepath.Base(res.ConfigFile))
	fmt.Printf("tensors: %d quantised, %d passthrough of %d\n", res.QuantizedWeights, res.PassthroughCount, res.TensorCount)
	fmt.Printf("bytes: %d -> %d\n", res.SourceBytes, res.OutputBytes)
}

// ggufFormatForBits maps a bit-width onto the closest named GGUF recipe.
// GGUF has no raw-bits knob (see gguf.QuantizeOptions.Format); `lem quant`'s
// own --gguf flag takes the recipe name directly (q4_k_m, q8_0, ...).
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
