// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference"
	pack "dappco.re/go/inference/model/pack"
)

// runPackCommand wires the engine-neutral .model container library
// (dappco.re/go/inference/model/pack) as a thin CLI: build, inspect, list,
// extract, and hash the Trix "MDL1" container that carries a model pack plus
// its manifest. No weights are loaded and no engine is touched — the whole
// verb is flag parsing + one library call per sub-action.
//
//	lem pack inspect model.model
//	lem pack create ~/models/gemma-4-e2b-it-4bit gemma.model --arch gemma4 --quant 4
func runPackCommand(_ context.Context, args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		printPackUsage(stderr)
		return 2
	}
	switch args[0] {
	case "create":
		return runPackCreate(args[1:], stdout, stderr)
	case "inspect":
		return runPackInspect(args[1:], stdout, stderr)
	case "list":
		return runPackList(args[1:], stdout, stderr)
	case "extract":
		return runPackExtract(args[1:], stdout, stderr)
	case "hash":
		return runPackHash(args[1:], stdout, stderr)
	case "-h", "--help", "help":
		printPackUsage(stdout)
		return 0
	default:
		core.Print(stderr, "%s pack: unknown subcommand %q", cliName(), args[0])
		printPackUsage(stderr)
		return 2
	}
}

func printPackUsage(w io.Writer) {
	name := cliName()
	core.WriteString(w, core.Sprintf("Usage: %s pack <subcommand> [flags]\n", name))
	core.WriteString(w, "\n")
	core.WriteString(w, "Build and read .model containers (Trix \"MDL1\") without loading weights.\n")
	core.WriteString(w, "\n")
	core.WriteString(w, "Subcommands\n")
	core.WriteString(w, "  create   <src-dir> <out.model>    pack a model directory into a .model container\n")
	core.WriteString(w, "  inspect  <file.model>             print the container manifest (no extraction)\n")
	core.WriteString(w, "  list     <file.model>             list the payload entries (path + size)\n")
	core.WriteString(w, "  extract  <file.model> <dest-dir>  unpack the container back to a directory\n")
	core.WriteString(w, "  hash     <src-dir>                print the canonical model-pack hash of a directory\n")
	core.WriteString(w, "\n")
	core.WriteString(w, core.Sprintf("Run \"%s pack <subcommand> --help\" for sub-action flags.\n", name))
}

// packSubUsage returns a Usage function that prints a synopsis, an optional
// description, and the sub-action's flags in the same shape the other lem
// verbs use.
func packSubUsage(fs *flag.FlagSet, w io.Writer, synopsis, desc string) func() {
	return func() {
		core.WriteString(w, core.Sprintf("Usage: %s %s\n", cliName(), synopsis))
		core.WriteString(w, "\n")
		if desc != "" {
			core.WriteString(w, desc)
			core.WriteString(w, "\n\n")
		}
		core.WriteString(w, "Flags:\n")
		printFlagBlock(w, fs)
	}
}

func runPackCreate(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("pack create"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	arch := fs.String("arch", "", "model architecture id recorded in the manifest")
	quant := fs.Int("quant", 0, "quantization bits recorded in the manifest")
	sourceFormat := fs.String("source-format", "safetensors", "on-disk weight format inside the payload: safetensors or gguf")
	producer := fs.String("producer", "lem", "producer name recorded in the manifest")
	fs.Usage = packSubUsage(fs, stderr, "pack create [flags] <src-dir> <out.model>",
		"Pack a model directory into a .model Trix container. The payload is a\n"+
			"deterministic tar of the directory; the manifest is embedded as the header.\n"+
			"Model identity is taken from the flags — no directory scan populates it.")
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 2 {
		core.Print(stderr, "%s pack create: expected <src-dir> <out.model>", cliName())
		fs.Usage()
		return 2
	}
	manifest := pack.Manifest{
		Model:        inference.ModelIdentity{Architecture: *arch, QuantBits: *quant},
		SourceFormat: *sourceFormat,
		Producer:     pack.Producer{Name: *producer},
	}
	if r := pack.Pack(fs.Arg(0), fs.Arg(1), pack.PackOptions{Manifest: manifest}); !r.OK {
		core.Print(stderr, "%s pack create: %s", cliName(), r.Error())
		return 1
	}
	core.Print(stdout, "wrote %s", fs.Arg(1))
	return 0
}

func runPackInspect(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("pack inspect"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print the manifest as JSON")
	fs.Usage = packSubUsage(fs, stderr, "pack inspect [flags] <file.model>",
		"Read a .model container header (no payload extraction) and print its manifest:\n"+
			"model identity, tokenizer, source format, producer, and capabilities.")
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.Print(stderr, "%s pack inspect: expected exactly one .model path", cliName())
		fs.Usage()
		return 2
	}
	manifest, _, r := pack.Inspect(fs.Arg(0))
	if !r.OK {
		core.Print(stderr, "%s pack inspect: %s", cliName(), r.Error())
		return 1
	}
	if *jsonOut {
		data := core.JSONMarshal(manifest)
		if !data.OK {
			core.Print(stderr, "%s pack inspect: %s", cliName(), data.Error())
			return 1
		}
		core.WriteString(stdout, string(data.Bytes()))
		core.WriteString(stdout, "\n")
		return 0
	}
	core.Print(stdout, "architecture:  %s", nonEmpty(manifest.Model.Architecture, "(unknown)"))
	core.Print(stdout, "source format: %s", nonEmpty(manifest.SourceFormat, "(unknown)"))
	core.Print(stdout, "quant bits:    %d", manifest.Model.QuantBits)
	core.Print(stdout, "context:       %d", manifest.Model.ContextLength)
	core.Print(stdout, "tokenizer:     %s", nonEmpty(manifest.Tokenizer.Kind, "(none)"))
	core.Print(stdout, "producer:      %s (%s)", nonEmpty(manifest.Producer.Name, "(unknown)"), manifest.Producer.Created)
	core.Print(stdout, "capabilities:  %d", len(manifest.Capabilities))
	return 0
}

func runPackList(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("pack list"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print the entries as JSON")
	fs.Usage = packSubUsage(fs, stderr, "pack list [flags] <file.model>",
		"List the payload tar entries of a .model container (path, size) without\n"+
			"extracting file contents.")
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.Print(stderr, "%s pack list: expected exactly one .model path", cliName())
		fs.Usage()
		return 2
	}
	entries, _, r := pack.List(fs.Arg(0))
	if !r.OK {
		core.Print(stderr, "%s pack list: %s", cliName(), r.Error())
		return 1
	}
	if *jsonOut {
		data := core.JSONMarshal(entries)
		if !data.OK {
			core.Print(stderr, "%s pack list: %s", cliName(), data.Error())
			return 1
		}
		core.WriteString(stdout, string(data.Bytes()))
		core.WriteString(stdout, "\n")
		return 0
	}
	for _, e := range entries {
		core.Print(stdout, "%12d  %s", e.Size, e.Path)
	}
	return 0
}

func runPackExtract(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("pack extract"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	overwrite := fs.Bool("overwrite", false, "allow extraction into a non-empty destination directory")
	fs.Usage = packSubUsage(fs, stderr, "pack extract [flags] <file.model> <dest-dir>",
		"Unpack a .model container's payload into a directory. Refuses a non-empty\n"+
			"destination unless --overwrite is set.")
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 2 {
		core.Print(stderr, "%s pack extract: expected <file.model> <dest-dir>", cliName())
		fs.Usage()
		return 2
	}
	if r := pack.Unpack(fs.Arg(0), fs.Arg(1), pack.UnpackOptions{Overwrite: *overwrite}); !r.OK {
		core.Print(stderr, "%s pack extract: %s", cliName(), r.Error())
		return 1
	}
	core.Print(stdout, "extracted %s -> %s", fs.Arg(0), fs.Arg(1))
	return 0
}

func runPackHash(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("pack hash"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = packSubUsage(fs, stderr, "pack hash <src-dir>",
		"Print the canonical model-pack hash of an unwrapped model directory — the\n"+
			"same value Pack embeds as Manifest.Model.Hash. Reads metadata files and\n"+
			"safetensors sizes only; does not read tensor bytes.")
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 1 {
		core.Print(stderr, "%s pack hash: expected exactly one directory", cliName())
		fs.Usage()
		return 2
	}
	h, r := pack.Hash(fs.Arg(0))
	if !r.OK {
		core.Print(stderr, "%s pack hash: %s", cliName(), r.Error())
		return 1
	}
	core.Print(stdout, "%s", h)
	return 0
}

// nonEmpty returns value when it is non-empty, otherwise fallback. Keeps the
// inspect summary readable when a manifest field is absent.
func nonEmpty(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}
