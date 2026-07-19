// SPDX-Licence-Identifier: EUPL-1.2

// TurboQuant live KV cache versus the native bf16 cache: the same model, the
// same greedy prompt, loaded twice — once default and once with
// inference.WithCacheMode("turboquant:3.5") (K codes at 4 bits, V at 3; the
// engine's global attention layers hold packed Lloyd-Max centroid codes + one
// f32 norm per row per head instead of bf16 rows). The program prints each
// mode's decode tok/s and the global-layer KV bytes per token, computed from
// the checkpoint's own config.json — the residency win is the point: at head
// dim 512 a global row per KV head costs 1024 bf16 bytes native versus
// 256+192+8 = 456 code bytes under 3.5 (2.2×; turboquant:2 reaches 3.9×).
//
// The codes are lossy by design (top-1 agreement ~97% against mode-off on
// gemma-4 e2b, measured band in the engine's live gate); modes: turboquant
// (=3.5), turboquant:4, turboquant:3.5, turboquant:3, turboquant:2.
//
//	go run ./pkg/kvcache-turboquant -model ~/models/gemma-4-e2b-4bit
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"dappco.re/go/inference"
	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "model snapshot directory (config.json + *.safetensors)")
	prompt := flag.String("prompt", "The old lighthouse keeper counted the ships as they passed:", "text to continue greedily under both cache modes")
	mode := flag.String("kv-cache", "turboquant:3.5", "TurboQuant mode to compare against the native default")
	tokens := flag.Int("max-tokens", 64, "tokens to decode per mode")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a model snapshot directory")
		os.Exit(2)
	}

	native, nativeRate := run(*model, "", *prompt, *tokens)
	tq, tqRate := run(*model, *mode, *prompt, *tokens)

	nBytes, tBytes, globals := kvBytesPerToken(*model, *mode)
	fmt.Printf("\nnative      : %6.1f tok/s   global-layer KV %5d B/token (%d global layers)\n", nativeRate, nBytes, globals)
	fmt.Printf("%-12s: %6.1f tok/s   global-layer KV %5d B/token (%.1fx smaller)\n", *mode, tqRate, tBytes, float64(nBytes)/float64(tBytes))
	fmt.Printf("\nnative continuation      : %s\n", oneLine(native))
	fmt.Printf("%-12s continuation: %s\n", *mode, oneLine(tq))
}

// run loads the model under one cache mode and decodes greedily, returning the
// generated text and the decode rate (tokens over wall time — the load and
// prefill are excluded from the timed span).
func run(model, mode, prompt string, maxTokens int) (string, float64) {
	opts := []inference.LoadOption{}
	label := "native"
	if mode != "" {
		opts = append(opts, inference.WithCacheMode(mode))
		label = mode
	}
	r := inference.LoadModel(model, opts...)
	if !r.OK {
		fmt.Fprintf(os.Stderr, "load (%s): %v\n", label, r.Value)
		os.Exit(1)
	}
	m := r.Value.(inference.TextModel)
	defer m.Close()

	var reply strings.Builder
	n := 0
	var started time.Time
	for tok := range m.Generate(context.Background(), prompt, inference.WithMaxTokens(maxTokens), inference.WithTemperature(0)) {
		if n == 0 {
			started = time.Now() // first token: prefill done, the decode clock starts
		}
		n++
		reply.WriteString(tok.Text)
	}
	if er := m.Err(); !er.OK {
		fmt.Fprintf(os.Stderr, "generate (%s): %v\n", label, er.Value)
		os.Exit(1)
	}
	rate := 0.0
	if n > 1 {
		rate = float64(n-1) / time.Since(started).Seconds()
	}
	return reply.String(), rate
}

// kvBytesPerToken reads the checkpoint's config.json and totals the GLOBAL
// attention layers' per-token KV cost under the native bf16 cache versus the
// TurboQuant mode: native = kvHeads·headDim·2 bytes per side; TurboQuant =
// kvHeads·(ceil(headDim·bits/8) + 4) per side (packed codes + the f32 norm).
func kvBytesPerToken(model, mode string) (nativeBytes, tqBytes, globals int) {
	kBits, vBits := 4, 3
	switch {
	case strings.HasSuffix(mode, ":4"):
		kBits, vBits = 4, 4
	case strings.HasSuffix(mode, ":3"):
		kBits, vBits = 3, 3
	case strings.HasSuffix(mode, ":2"):
		kBits, vBits = 2, 2
	}
	raw, err := os.ReadFile(filepath.Join(model, "config.json"))
	if err != nil {
		return 0, 0, 0
	}
	var cfg struct {
		TextConfig *struct {
			HeadDim       int      `json:"head_dim"`
			GlobalHeadDim int      `json:"global_head_dim"`
			KVHeads       int      `json:"num_key_value_heads"`
			LayerTypes    []string `json:"layer_types"`
		} `json:"text_config"`
		HeadDim       int      `json:"head_dim"`
		GlobalHeadDim int      `json:"global_head_dim"`
		KVHeads       int      `json:"num_key_value_heads"`
		LayerTypes    []string `json:"layer_types"`
	}
	if json.Unmarshal(raw, &cfg) != nil {
		return 0, 0, 0
	}
	hd, ghd, kv, types := cfg.HeadDim, cfg.GlobalHeadDim, cfg.KVHeads, cfg.LayerTypes
	if cfg.TextConfig != nil {
		hd, ghd, kv, types = cfg.TextConfig.HeadDim, cfg.TextConfig.GlobalHeadDim, cfg.TextConfig.KVHeads, cfg.TextConfig.LayerTypes
	}
	if ghd == 0 {
		ghd = hd
	}
	if kv == 0 {
		kv = 1
	}
	for _, lt := range types {
		if lt != "full_attention" {
			continue
		}
		globals++
		nativeBytes += kv * ghd * 2 * 2 // bf16 K + V
		tqBytes += kv * ((ghd*kBits+7)/8 + 4 + (ghd*vBits+7)/8 + 4)
	}
	return nativeBytes, tqBytes, globals
}

// oneLine collapses a continuation for the side-by-side print.
func oneLine(s string) string {
	s = strings.ReplaceAll(s, "\n", "\\n")
	if len(s) > 140 {
		return s[:140] + "…"
	}
	return s
}
