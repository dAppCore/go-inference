// SPDX-Licence-Identifier: EUPL-1.2

// Command lem is Lethean's sovereign inference binary: it hosts an
// OpenAI/Anthropic/Ollama-compatible HTTP API for a local model, compiled from
// go-inference alone (no go-mlx). Each subcommand is thin flag-parsing that
// wires a go-inference library — the serve business logic lives in
// dappco.re/go/inference/serving, not here.
//
//	lem serve --model ~/models/gemma-4-e2b-it-4bit --addr :36911
package main

import (
	"context"
	"io"
	"os/signal"
	"syscall"

	core "dappco.re/go"

	_ "dappco.re/go/inference/engine/metal"  // registers the no-cgo Apple "metal" backend via init()
	_ "dappco.re/go/inference/model/builtin" // registers the built-in arches (gemma3/gemma4/mistral/qwen3)
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	args := core.Args()
	if len(args) > 0 {
		if name := core.PathBase(args[0]); name != "" {
			commandName = name
		}
	}
	core.Exit(runCommand(ctx, args[1:], core.Stdout(), core.Stderr()))
}

// commandName is the invoked binary name (argv[0] base), used in usage + notice
// lines so `lem` and any renamed copy print their own name.
var commandName = "lem"

func cliName() string {
	name := core.Trim(commandName)
	if name == "" {
		return "lem"
	}
	return name
}

func cliCommandName(command string) string {
	if command == "" {
		return cliName()
	}
	return cliName() + " " + command
}

func runCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		printUsage(stdout)
		return 0
	}
	switch args[0] {
	case "serve":
		return runServeCommand(ctx, args[1:], stdout, stderr)
	case "generate":
		return runGenerateCommand(ctx, args[1:], stdout, stderr)
	case "ssd":
		return runSSDCommand(ctx, args[1:], stdout, stderr)
	case "sft":
		return runSFTCommand(ctx, args[1:], stdout, stderr)
	case "tune":
		return runTuneCommand(ctx, args[1:], stdout, stderr)
	case "-h", "--help", "help":
		printUsage(stdout)
		return 0
	default:
		core.Print(stderr, "%s: unknown command %q", cliName(), args[0])
		printUsage(stderr)
		return 2
	}
}

func printUsage(w io.Writer) {
	name := cliName()
	core.WriteString(w, core.Sprintf("Usage: %s <command> [flags]\n", name))
	core.WriteString(w, "\n")
	core.WriteString(w, "Run inference\n")
	core.WriteString(w, "  serve               host OpenAI/Anthropic/Ollama HTTP API for a loaded model\n")
	core.WriteString(w, "  generate            one-shot generate + decode tok/s (no serve; like-for-like bench)\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "Train\n")
	core.WriteString(w, "  ssd                 self-distillation sampling: sample the frozen base, capture the trace\n")
	core.WriteString(w, "  sft                 LoRA supervised fine-tuning through the engine trainer seam\n")
	core.WriteString(w, "  tune                measure + persist the best MTP draft block as a serve profile\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "Examples\n")
	core.WriteString(w, core.Sprintf("  %s serve --model ~/models/gemma-4-e2b-it-4bit         # OpenAI HTTP on :36911\n", name))
	core.WriteString(w, core.Sprintf("  %s serve --model ~/models/gemma-4-e2b-it-4bit --context 8192\n", name))
	core.WriteString(w, "\n")
	core.WriteString(w, core.Sprintf("Run \"%s <command> -h\" for command-specific flags.\n", name))
}
