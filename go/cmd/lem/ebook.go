// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"flag"
	"io"

	core "dappco.re/go"
	"dappco.re/go/inference/modelmgmt"
	coreio "dappco.re/go/io"
)

// runEbookCommand renders a model directory into a valid EPUB3: the authored
// foreword (README — the human-speech anchor), a method section, and — by
// default — the weights as base64 plates that decode back into a runnable
// model. The point is the PGP playbook: a published, authored book carries the
// protection of speech; only a court can strip it. Pure file I/O — no model is
// loaded, so it is engine-neutral. Thin: flag parsing + one library call. The
// EPUB render engine lives in dappco.re/go/inference/modelmgmt, not here.
//
//	lem ebook --model ~/Code/lthn/LEM-Gemma3-1B --out LEM-Gemma3-1B.epub
//	lem ebook --model <dir> --weights=false   # the readable manifesto, no plates
func runEbookCommand(_ context.Context, args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("ebook"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	modelDir := fs.String("model", "", "model directory to render (required)")
	out := fs.String("out", "", "output .epub path (default <model-name>.epub in the working dir)")
	title := fs.String("title", "", "book title (default the model directory name)")
	author := fs.String("author", "Lethean", "book author — the publishing voice that makes it authored speech")
	foreword := fs.String("foreword", "", "foreword text file (default <model>/README.md when present)")
	weights := fs.Bool("weights", true, "include the weights as base64 plates (the reconstructable artifact); false = manifesto + method only")
	chapterChars := fs.Int("chapter-chars", 0, "base64 characters per weight plate (0 = default 4,000,000)")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s ebook --model <dir> [--out book.epub] [flags]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Render a model as an EPUB3 book: authored foreword + method + the weights as\n")
		core.WriteString(stderr, "base64 plates that decode back into a runnable model. EUPL-1.2. A published\n")
		core.WriteString(stderr, "book is protected speech — the PGP playbook, applied to weights.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Example:\n")
		core.WriteString(stderr, core.Sprintf("  %s ebook --model ~/Code/lthn/LEM-Gemma3-1B --out LEM-Gemma3-1B.epub\n", name))
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if *modelDir == "" {
		fs.Usage()
		return 2
	}

	outPath := *out
	if outPath == "" {
		outPath = core.PathBase(*modelDir) + ".epub"
	}

	built := modelmgmt.BuildModelBook(modelmgmt.ModelBookOptions{
		ModelDir:        *modelDir,
		Title:           *title,
		Author:          *author,
		ForewordPath:    *foreword,
		IncludeWeights:  *weights,
		ChapterChars:    *chapterChars,
		GeneratorCredit: "lem ebook",
	})
	if !built.OK {
		core.Print(stderr, "%s ebook: %s", cliName(), built.Error())
		return 1
	}
	book := built.Value.(*modelmgmt.Book)

	w, err := coreio.Local.Create(outPath)
	if err != nil {
		core.Print(stderr, "%s ebook: create %s: %v", cliName(), outPath, err)
		return 1
	}
	if wr := book.WriteEPUB(w); !wr.OK {
		_ = w.Close()
		core.Print(stderr, "%s ebook: %s", cliName(), wr.Error())
		return 1
	}
	if cerr := w.Close(); cerr != nil {
		core.Print(stderr, "%s ebook: close %s: %v", cliName(), outPath, cerr)
		return 1
	}

	navChapters := 0
	for i := range book.Chapters {
		if book.Chapters[i].InNav {
			navChapters++
		}
	}
	core.Print(stdout, "wrote %s — %d chapters (%d in contents)", outPath, len(book.Chapters), navChapters)
	if info, serr := coreio.Local.Stat(outPath); serr == nil {
		core.Print(stdout, "epub size %d bytes", info.Size())
	}
	return 0
}
