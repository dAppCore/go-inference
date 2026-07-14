// SPDX-Licence-Identifier: EUPL-1.2

// Command lthn-model-pack wraps the model/pack primitives as a CLI so
// .model Trix containers can be built, extracted, and inspected from the
// terminal without going through a service.
//
//	lthn-model-pack pack /models/gemma-3-4b-it /out/gemma-3-4b-it.model --arch gemma --quant 4
//	lthn-model-pack inspect /out/gemma-3-4b-it.model
//	lthn-model-pack unpack  /out/gemma-3-4b-it.model /tmp/extracted
package main

import (
	"flag"
	"os"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/model/pack"
)

const usage = `Usage:
  lthn-model-pack pack    [--arch X] [--quant N] [--source safetensors|gguf] [--producer X] <srcDir> <out.model>
  lthn-model-pack unpack  [--overwrite]                                                      <src.model> <destDir>
  lthn-model-pack list                                                                       <src.model>
  lthn-model-pack inspect                                                                    <src.model>

Flags must come before positional arguments.`

func main() {
	if len(os.Args) < 2 {
		core.Print(os.Stderr, "%s", usage)
		os.Exit(2)
	}
	var r core.Result
	switch os.Args[1] {
	case "pack":
		r = runPack(os.Args[2:])
	case "unpack":
		r = runUnpack(os.Args[2:])
	case "list":
		r = runList(os.Args[2:])
	case "inspect":
		r = runInspect(os.Args[2:])
	case "-h", "--help", "help":
		core.Print(os.Stdout, "%s", usage)
		return
	default:
		core.Print(os.Stderr, "unknown verb %q", os.Args[1])
		core.Print(os.Stderr, "%s", usage)
		os.Exit(2)
	}
	if !r.OK {
		core.Print(os.Stderr, "lthn-model-pack: %v", r.Value)
		os.Exit(1)
	}
}

func runPack(args []string) core.Result {
	fs := flag.NewFlagSet("pack", flag.ExitOnError)
	arch := fs.String("arch", "", "model architecture (e.g. gemma)")
	quantBits := fs.Int("quant", 0, "quantisation bits (0 for none)")
	sourceFormat := fs.String("source", "safetensors", "source format: safetensors|gguf")
	producerName := fs.String("producer", "lthn-model-pack", "producer name")
	if err := fs.Parse(args); err != nil {
		return core.Fail(core.E("pack", "parse flags", err))
	}
	rest := fs.Args()
	if len(rest) != 2 {
		return core.Fail(core.E("pack", "expected: pack <srcDir> <out.model>", nil))
	}
	srcDir, dest := rest[0], rest[1]

	r := pack.Pack(srcDir, dest, pack.PackOptions{
		Manifest: pack.Manifest{
			Model: inference.ModelIdentity{
				Architecture: *arch,
				QuantBits:    *quantBits,
			},
			SourceFormat: *sourceFormat,
			Producer:     pack.Producer{Name: *producerName},
		},
	})
	if r.OK {
		core.Print(os.Stdout, "packed %s -> %s", srcDir, dest)
	}
	return r
}

func runUnpack(args []string) core.Result {
	fs := flag.NewFlagSet("unpack", flag.ExitOnError)
	overwrite := fs.Bool("overwrite", false, "allow writing into a non-empty destDir")
	if err := fs.Parse(args); err != nil {
		return core.Fail(core.E("unpack", "parse flags", err))
	}
	rest := fs.Args()
	if len(rest) != 2 {
		return core.Fail(core.E("unpack", "expected: unpack <src.model> <destDir>", nil))
	}
	src, destDir := rest[0], rest[1]

	r := pack.Unpack(src, destDir, pack.UnpackOptions{Overwrite: *overwrite})
	if r.OK {
		core.Print(os.Stdout, "unpacked %s -> %s", src, destDir)
	}
	return r
}

func runList(args []string) core.Result {
	if len(args) != 1 {
		return core.Fail(core.E("list", "expected: list <src.model>", nil))
	}
	src := args[0]

	entries, manifest, r := pack.List(src)
	if !r.OK {
		return r
	}
	bundle := map[string]any{
		"manifest": manifest,
		"entries":  entries,
		"count":    len(entries),
	}
	jr := core.JSONMarshalIndent(bundle, "", "  ")
	if !jr.OK {
		return jr
	}
	core.Print(os.Stdout, "%s", string(jr.Value.([]byte)))
	return core.Ok(nil)
}

func runInspect(args []string) core.Result {
	if len(args) != 1 {
		return core.Fail(core.E("inspect", "expected: inspect <src.model>", nil))
	}
	src := args[0]

	manifest, inspection, r := pack.Inspect(src)
	if !r.OK {
		return r
	}
	bundle := map[string]any{
		"manifest":    manifest,
		"inspection":  inspection,
		"fingerprint": pack.Fingerprint(*manifest),
	}
	jr := core.JSONMarshalIndent(bundle, "", "  ")
	if !jr.OK {
		return jr
	}
	core.Print(os.Stdout, "%s", string(jr.Value.([]byte)))
	return core.Ok(nil)
}
