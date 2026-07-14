// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/gguf"
	"dappco.re/go/inference/model/quant/awq"
	"dappco.re/go/inference/model/quant/fp8"
	"dappco.re/go/inference/model/quant/gptq"
	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/quant/nf4"
)

// runQuantCommand wires the model quantisers as a thin CLI: hand it a dense (bf16/f32)
// safetensors model directory and it writes a quantised model directory. Two lanes:
//
//   - default — MLX group-affine (mlxaffine.ConvertSnapshot): the packed-uint32 +
//     bf16 scales/biases format the engine loads natively. Byte-for-byte what
//     mlx_lm.convert produces (gated in the mlxaffine byte-oracle).
//
//   - -gguf <FORMAT> — the existing GGUF whole-model pipeline
//     (gguf.QuantizeModelPack): q4_k_m, q8_0, q5_k, q6_k, …
//
//     lem quant ~/models/gemma-4-12B-it-bf16                       # → …-4bit (MLX affine)
//     lem quant ~/models/gemma-4-12B-it-bf16 -bits 8 -group-size 32
//     lem quant ~/models/gemma-4-12B-it-bf16 -gguf q4_k_m          # → …-gguf-q4_k_m
func runQuantCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("quant"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	bits := fs.Int("bits", 4, "affine quantisation bit-width (2, 4, or 8) — MLX lane")
	groupSize := fs.Int("group-size", 64, "affine quantisation group size — MLX lane")
	out := fs.String("o", "", "output model directory (default: <src>-<bits>bit, or <src>-gguf-<format>)")
	ggufFormat := fs.String("gguf", "", "instead of MLX affine, run the GGUF lane in this format (q4_k_m, q8_0, q5_k, q6_k, …)")
	gptqFormat := fs.Bool("gptq", false, "instead of MLX affine, write HF GPTQ qweight/qzeros/scales/g_idx tensors")
	awqFormat := fs.Bool("awq", false, "instead of MLX affine, write HF AutoAWQ GEMM qweight/qzeros/scales tensors (data-free approximation)")
	fp8Format := fs.Bool("fp8", false, "instead of MLX affine, write compressed-tensors static per-tensor E4M3 weights")
	nf4Format := fs.Bool("nf4", false, "instead of MLX affine, write bitsandbytes NF4 blockwise weights")
	fs.Usage = quantUsage(fs, stderr)
	// Two-phase parse so the <src-model-dir> positional may appear before OR after the
	// flags (Go's flag stops at the first non-flag): the first Parse consumes any
	// leading flags and stops at the positional; a second Parse over the remainder picks
	// up trailing flags. Matches the documented `quant <src> [-flags]` shape.
	src, err := parseWithPositional(fs, args)
	if err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		if src == "" {
			core.Print(stderr, "%s quant: expected exactly one <src-model-dir>", cliName())
		} else {
			core.Print(stderr, "%s quant: %s", cliName(), err.Error())
		}
		fs.Usage()
		return 2
	}
	if !dirHasSafetensors(src) {
		core.Print(stderr, "%s quant: %s has no *.safetensors shards", cliName(), src)
		return 1
	}
	formats := 0
	for _, selected := range []bool{*gptqFormat, *awqFormat, *fp8Format, *nf4Format, core.Trim(*ggufFormat) != ""} {
		if selected {
			formats++
		}
	}
	if formats > 1 {
		core.Print(stderr, "%s quant: -gptq, -awq, -fp8, -nf4, and -gguf are mutually exclusive", cliName())
		return 2
	}

	if core.Trim(*ggufFormat) != "" {
		return runQuantGGUF(ctx, src, *out, *ggufFormat, stdout, stderr)
	}
	if *gptqFormat {
		return runQuantGPTQ(ctx, src, *out, *bits, *groupSize, stdout, stderr)
	}
	if *awqFormat {
		return runQuantAWQ(ctx, src, *out, *bits, *groupSize, stdout, stderr)
	}
	if *fp8Format {
		return runQuantFP8(ctx, src, *out, stdout, stderr)
	}
	if *nf4Format {
		return runQuantNF4(ctx, src, *out, stdout, stderr)
	}
	return runQuantMLXAffine(ctx, src, *out, *bits, *groupSize, stdout, stderr)
}

func runQuantAWQ(ctx context.Context, src, out string, bits, groupSize int, stdout, stderr io.Writer) int {
	outDir := out
	if core.Trim(outDir) == "" {
		outDir = defaultOutDir(src, core.Sprintf("awq-%dbit", bits))
	}
	core.Print(stdout, "quant (AWQ-compatible data-free zero-point %d-bit, group %d): %s -> %s", bits, groupSize, src, outDir)
	core.Print(stderr, "note: no calibration data supplied; writing a data-free weight-only approximation, not true activation-aware AWQ")
	progress := func(name string, quantised bool, index, total int) {
		verb := "copy    "
		if quantised {
			verb = "quantise"
		}
		core.Print(stderr, "[%4d/%4d] %s  %s", index, total, verb, name)
	}
	result, err := awq.ConvertSnapshot(ctx, src, outDir, awq.Options{Bits: bits, GroupSize: groupSize, ZeroPoint: true}, progress)
	if err != nil {
		core.Print(stderr, "%s quant: %s", cliName(), err.Error())
		return 1
	}
	core.Print(stdout, "done: %d tensors (%d quantised, %d passed through), %s in -> %s out", result.TensorCount, result.QuantizedWeights, result.PassthroughCount, humanBytes(result.SourceBytes), humanBytes(result.OutputBytes))
	core.Print(stdout, "wrote %s", result.WeightFile)
	return 0
}

func runQuantFP8(ctx context.Context, src, out string, stdout, stderr io.Writer) int {
	if core.Trim(out) == "" {
		out = defaultOutDir(src, "fp8")
	}
	core.Print(stdout, "quant (compressed-tensors FP8 E4M3): %s -> %s", src, out)
	result, err := fp8.ConvertSnapshot(ctx, src, out, quantProgress(stderr))
	if err != nil {
		core.Print(stderr, "%s quant: %s", cliName(), err.Error())
		return 1
	}
	core.Print(stdout, "done: %d tensors (%d quantised, %d passed through)", result.TensorCount, result.QuantizedWeights, result.PassthroughCount)
	core.Print(stdout, "wrote %s", result.WeightFile)
	return 0
}

func runQuantNF4(ctx context.Context, src, out string, stdout, stderr io.Writer) int {
	if core.Trim(out) == "" {
		out = defaultOutDir(src, "nf4")
	}
	core.Print(stdout, "quant (bitsandbytes NF4, block 64, no double quantisation): %s -> %s", src, out)
	result, err := nf4.ConvertSnapshot(ctx, src, out, quantProgress(stderr))
	if err != nil {
		core.Print(stderr, "%s quant: %s", cliName(), err.Error())
		return 1
	}
	core.Print(stdout, "done: %d tensors (%d quantised, %d passed through)", result.TensorCount, result.QuantizedWeights, result.PassthroughCount)
	core.Print(stdout, "wrote %s", result.WeightFile)
	return 0
}

func quantProgress(stderr io.Writer) func(string, bool, int, int) {
	return func(name string, quantised bool, index, total int) {
		verb := "copy    "
		if quantised {
			verb = "quantise"
		}
		core.Print(stderr, "[%4d/%4d] %s  %s", index, total, verb, name)
	}
}

func runQuantGPTQ(ctx context.Context, src, out string, bits, groupSize int, stdout, stderr io.Writer) int {
	outDir := out
	if core.Trim(outDir) == "" {
		outDir = defaultOutDir(src, core.Sprintf("gptq-%dbit", bits))
	}
	core.Print(stdout, "quant (GPTQ %d-bit, group %d): %s -> %s", bits, groupSize, src, outDir)
	progress := func(name string, quantised bool, index, total int) {
		verb := "copy    "
		if quantised {
			verb = "quantise"
		}
		core.Print(stderr, "[%4d/%4d] %s  %s", index, total, verb, name)
	}
	result, err := gptq.ConvertSnapshot(ctx, src, outDir, gptq.Options{Bits: bits, GroupSize: groupSize, Symmetric: true}, progress)
	if err != nil {
		core.Print(stderr, "%s quant: %s", cliName(), err.Error())
		return 1
	}
	core.Print(stdout, "done: %d tensors (%d quantised, %d passed through), %s in -> %s out", result.TensorCount, result.QuantizedWeights, result.PassthroughCount, humanBytes(result.SourceBytes), humanBytes(result.OutputBytes))
	core.Print(stdout, "wrote %s", result.WeightFile)
	return 0
}

// parseWithPositional parses fs allowing the single positional argument to sit before
// or after the flags. It returns the positional and an error; a nil error guarantees
// exactly one positional was found and no flags were left dangling. flag.ErrHelp is
// propagated so the caller can exit 0 on -h.
func parseWithPositional(fs *flag.FlagSet, args []string) (string, error) {
	if err := fs.Parse(args); err != nil {
		return "", err
	}
	if fs.NArg() == 0 {
		return "", core.NewError("missing <src-model-dir>")
	}
	positional := fs.Arg(0)
	rest := fs.Args()[1:]
	if len(rest) > 0 {
		if err := fs.Parse(rest); err != nil {
			return positional, err
		}
		if fs.NArg() != 0 {
			return positional, core.Errorf("unexpected extra argument %q", fs.Arg(0))
		}
	}
	return positional, nil
}

func quantUsage(fs *flag.FlagSet, w io.Writer) func() {
	return func() {
		core.WriteString(w, core.Sprintf("Usage: %s quant [flags] <src-model-dir>\n\n", cliName()))
		core.WriteString(w, "Quantise a dense (bf16/f32) safetensors model directory into a quantised model\n")
		core.WriteString(w, "directory. Default lane writes the MLX group-affine format the engine loads\n")
		core.WriteString(w, "natively; -gguf switches to the GGUF whole-model pipeline.\n\n")
		core.WriteString(w, "Flags:\n")
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(w, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(w, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
}

// runQuantMLXAffine runs the MLX group-affine lane with a per-tensor progress line and a
// final before→after summary.
func runQuantMLXAffine(ctx context.Context, src, out string, bits, groupSize int, stdout, stderr io.Writer) int {
	if !mlxaffine.SupportedBits(bits) {
		core.Print(stderr, "%s quant: unsupported -bits %d (want 2, 4, or 8)", cliName(), bits)
		return 1
	}
	outDir := out
	if core.Trim(outDir) == "" {
		outDir = defaultOutDir(src, core.Sprintf("%dbit", bits))
	}
	core.Print(stdout, "quant (MLX affine %d-bit, group %d): %s -> %s", bits, groupSize, src, outDir)

	progress := func(name string, quantised bool, index, total int) {
		verb := "copy    "
		if quantised {
			verb = "quantise"
		}
		core.Print(stderr, "[%4d/%4d] %s  %s", index, total, verb, name)
	}
	res, err := mlxaffine.ConvertSnapshot(ctx, src, outDir, mlxaffine.Options{Bits: bits, GroupSize: groupSize}, progress)
	if err != nil {
		core.Print(stderr, "%s quant: %s", cliName(), err.Error())
		return 1
	}
	core.Print(stdout, "done: %d tensors (%d quantised, %d passed through), %s in -> %s out",
		res.TensorCount, res.QuantizedWeights, res.PassthroughCount, humanBytes(res.SourceBytes), humanBytes(res.OutputBytes))
	core.Print(stdout, "wrote %s", res.WeightFile)
	return 0
}

// runQuantGGUF runs the existing GGUF whole-model quantisation pipeline.
func runQuantGGUF(ctx context.Context, src, out, format string, stdout, stderr io.Writer) int {
	outDir := out
	if core.Trim(outDir) == "" {
		outDir = defaultOutDir(src, "gguf-"+core.Lower(core.Trim(format)))
	}
	weightFiles := core.PathGlob(core.PathJoin(src, "*.safetensors"))
	core.SliceSort(weightFiles)
	arch, _, _ := model.ProbeDirArch(src) // best-effort architecture for GGUF metadata

	core.Print(stdout, "quant (GGUF %s): %s -> %s", core.Lower(core.Trim(format)), src, outDir)
	res, err := gguf.QuantizeModelPack(ctx, gguf.QuantizeOptions{
		SourcePack: gguf.Source{Root: src, Architecture: arch, WeightFiles: weightFiles},
		OutputPath: outDir,
		Format:     gguf.QuantizeFormat(core.Trim(format)),
	})
	if err != nil {
		core.Print(stderr, "%s quant: %s", cliName(), err.Error())
		return 1
	}
	for _, note := range res.Notes {
		core.Print(stderr, "note: %s", note)
	}
	core.Print(stdout, "done: %d tensors quantised to %s", res.QuantizedTensors, res.Format)
	core.Print(stdout, "wrote %s", res.WeightPath)
	return 0
}

// defaultOutDir derives the output directory beside the source: the source's basename
// with a "-<suffix>" tag, in the source's parent directory (the mlx_lm.convert
// convention, e.g. gemma-4-12B-it-bf16 → gemma-4-12B-it-4bit when suffix is "4bit").
func defaultOutDir(src, suffix string) string {
	base := core.PathBase(core.TrimSuffix(src, "/"))
	base = core.TrimSuffix(base, "-bf16")
	base = core.TrimSuffix(base, "-f32")
	return core.PathJoin(core.PathDir(core.TrimSuffix(src, "/")), base+"-"+suffix)
}

// dirHasSafetensors reports whether dir holds at least one *.safetensors shard.
func dirHasSafetensors(dir string) bool {
	return len(core.PathGlob(core.PathJoin(dir, "*.safetensors"))) > 0
}

// humanBytes renders a byte count as a compact MiB/GiB string for the summary line.
func humanBytes(n int64) string {
	const (
		mib = 1 << 20
		gib = 1 << 30
	)
	switch {
	case n >= gib:
		return core.Sprintf("%.2f GiB", float64(n)/gib)
	case n >= mib:
		return core.Sprintf("%.1f MiB", float64(n)/mib)
	default:
		return core.Sprintf("%d B", n)
	}
}
