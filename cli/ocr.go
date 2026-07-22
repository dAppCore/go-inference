// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/arch/deepseek-ai/deepseekvl2"
	"dappco.re/go/inference/model/arch/rednote-hilab/dotsocr"
	"dappco.re/go/inference/model/arch/zai-org/glmocr"
)

// ocr.go is thin flag-parsing over dappco.re/go/inference/model/arch/deepseek-ai/deepseekvl2's
// Load/Model.OCR — the OCR business logic lives there, not here (mirroring generate.go's/
// transcribe.go's doc comments: "the business logic lives in the arch package, not here").
// DeepSeek-OCR's dual-tower vision encoder + MoE decoder never enters model.Assemble (see
// deepseekvl2.Config.Arch's doc comment), so this verb calls deepseekvl2.Load directly rather
// than going through the shared generate/serve model-loading path — the same "own loader" shape
// transcribe.go uses for Whisper.
//
//	lem ocr --model ~/models/deepseek-ocr page.png
//
// ctx carries no cancellation point yet — Model.OCR is a single host-CPU forward pass with no
// natural cut point (v1 scope; a later slice could thread it through the decode loop, mirroring
// transcribe.go's identical note).
func runOCRCommand(_ context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("ocr"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelDir := fs.String("model", "", "DeepSeek-OCR checkpoint directory (config.json, tokenizer.json, *.safetensors)")
	prompt := fs.String("prompt", "", "override the OCR prompt (must contain exactly one <image> placeholder); empty uses the checkpoint's recommended default")
	maxNewTokens := fs.Int("max-tokens", 0, "cap the generated content length; <=0 uses the package default")
	jsonOut := fs.Bool("json", false, "print {\"text\":...} instead of plain OCR text")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s ocr --model <deepseek-ocr-dir> [flags] <image.png>\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Run OCR on one image (PNG/JPEG) through a loaded DeepSeek-OCR checkpoint:\n")
		core.WriteString(stderr, "host-f32 greedy decode, the fixed 1024x1024 \"Base\" resolution mode (v1 scope —\n")
		core.WriteString(stderr, "any other image size is a named refusal, not a silent resize).\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		printFlagBlock(stderr, fs)
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Examples:\n")
		core.WriteString(stderr, core.Sprintf("  %s ocr --model ~/models/deepseek-ocr page.png\n", name))
		core.WriteString(stderr, "    # default \"Free OCR\" prompt, transcript to stdout\n")
		core.WriteString(stderr, core.Sprintf("  %s ocr --model ~/models/deepseek-ocr --json page.png\n", name))
		core.WriteString(stderr, "    # {\"text\":...} to stdout\n")
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if *modelDir == "" {
		core.Print(stderr, "%s ocr: --model is required", cliName())
		fs.Usage()
		return 2
	}
	if fs.NArg() != 1 {
		core.Print(stderr, "%s ocr: expected exactly one image path", cliName())
		fs.Usage()
		return 2
	}

	imagePath := fs.Arg(0)
	read := core.ReadFile(imagePath)
	if !read.OK {
		core.Print(stderr, "%s ocr: read %s: %s", cliName(), imagePath, read.Error())
		return 1
	}
	imageBytes, ok := read.Value.([]byte)
	if !ok {
		core.Print(stderr, "%s ocr: read %s returned non-byte data", cliName(), imagePath)
		return 1
	}

	// Dispatch on the checkpoint's declared model_type — one verb, every OCR arch.
	mt, _, perr := model.ProbeDirArch(*modelDir)
	if perr != nil {
		core.Print(stderr, "%s ocr: probe %s: %v", cliName(), *modelDir, perr)
		return 1
	}
	var text string
	var oerr error
	switch mt {
	case "deepseek_vl_v2":
		m, err := deepseekvl2.Load(*modelDir)
		if err != nil {
			core.Print(stderr, "%s ocr: %v", cliName(), err)
			return 1
		}
		result, err := m.OCR(imageBytes, deepseekvl2.Options{Prompt: *prompt, MaxNewTokens: *maxNewTokens})
		if err == nil {
			text = result.Text
		}
		oerr = err
	case "dots_ocr", "dots_ocr_1_5":
		m, err := dotsocr.Load(*modelDir)
		if err != nil {
			core.Print(stderr, "%s ocr: %v", cliName(), err)
			return 1
		}
		text, oerr = m.OCR(imageBytes, *prompt)
	case "glm_ocr", "glm_ocr_text":
		m, err := glmocr.Load(*modelDir)
		if err != nil {
			core.Print(stderr, "%s ocr: %v", cliName(), err)
			return 1
		}
		text, oerr = m.OCR(imageBytes, *prompt)
	default:
		core.Print(stderr, "%s ocr: model_type %q is not an OCR arch this verb serves (deepseek_vl_v2, dots_ocr, dots_ocr_1_5, glm_ocr, glm_ocr_text)", cliName(), mt)
		return 1
	}
	if oerr != nil {
		core.Print(stderr, "%s ocr: %v", cliName(), oerr)
		return 1
	}

	if *jsonOut {
		printJSON(stdout, map[string]string{"text": text})
		return 0
	}
	core.WriteString(stdout, text)
	core.WriteString(stdout, "\n")
	return 0
}
