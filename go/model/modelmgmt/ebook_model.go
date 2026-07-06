// SPDX-Licence-Identifier: EUPL-1.2

package modelmgmt

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"io"
	"sort"
	"time"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// ModelBookOptions configures BuildModelBook.
type ModelBookOptions struct {
	ModelDir       string
	Title          string // "" → the model directory's base name
	Author         string // "" → "Lethean"
	ForewordPath   string // "" → <ModelDir>/README.md when present
	IncludeWeights bool   // false → the authored manifesto + method only (a readable book, no plates)
	ChapterChars   int    // base64 chars per weight chapter; <=0 → default
	// GeneratorCredit names the tool credited in the colophon's "generated
	// by <code>…</code>" line. "" → defaultGeneratorCredit (this package's
	// own entry point); a non-empty value is used verbatim in that credit
	// position instead, escaped exactly like every other user-supplied
	// string in the book (title, author, foreword). Set this from a CLI
	// wrapper (e.g. lthn-mlx's "ebook" command) so the colophon credits the
	// command a reader actually ran, rather than the library call beneath
	// it.
	GeneratorCredit string
}

const (
	defaultWeightChapterChars = 4_000_000
	charsPerPrintedPage       = 2000
	pagesPerVolume            = 300

	// defaultGeneratorCredit is the colophon credit line used when
	// ModelBookOptions.GeneratorCredit is empty — this package's own entry
	// point, matching the credit BuildModelBook has always emitted.
	defaultGeneratorCredit = "modelmgmt.BuildModelBook"

	euplNotice = "This work is licensed under the European Union Public Licence v1.2 (EUPL-1.2). " +
		"You may use, study, share and modify it under the terms of that licence. " +
		"Full text: https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12"
)

// weightFile is one safetensors file rendered into the book.
type weightFile struct {
	name     string
	bytes    int
	sha256   string
	tensors  int
	elements int64
	b64      string // base64 of the whole file
}

// BuildModelBook reads a model directory and assembles it as an authored
// book: title + licence, the foreword (README — the human-speech anchor), the
// method section (architecture + inventory + this-book-in-numbers), and —
// when IncludeWeights is set — the weights as base64 plates plus a decode
// recipe so the book reconstructs into a runnable model. No model is loaded;
// this reads bytes and arranges them, so it carries no engine dependency and
// runs identically whichever backend (mlx, rocm, cpu) produced the weights.
//
//	r := modelmgmt.BuildModelBook(modelmgmt.ModelBookOptions{ModelDir: dir, IncludeWeights: true})
//	if !r.OK { return r }
//	book := r.Value.(*modelmgmt.Book)
func BuildModelBook(opts ModelBookOptions) core.Result {
	if opts.ModelDir == "" {
		return core.Fail(core.E("modelmgmt.BuildModelBook", "model dir is required", nil))
	}
	entries, err := coreio.Local.List(opts.ModelDir)
	if err != nil {
		return core.Fail(core.E("modelmgmt.BuildModelBook", "list model dir", err))
	}
	title := opts.Title
	if title == "" {
		title = core.PathBase(opts.ModelDir)
	}
	author := opts.Author
	if author == "" {
		author = "Lethean"
	}
	chapterChars := opts.ChapterChars
	if chapterChars <= 0 {
		chapterChars = defaultWeightChapterChars
	}

	// Foreword: explicit path, else README.md in the model dir, else a note.
	foreword := ""
	forewordPath := opts.ForewordPath
	if forewordPath == "" {
		candidate := core.JoinPath(opts.ModelDir, "README.md")
		if coreio.Local.IsFile(candidate) {
			forewordPath = candidate
		}
	}
	if forewordPath != "" {
		text, ferr := coreio.Local.Read(forewordPath)
		if ferr != nil {
			return core.Fail(core.E("modelmgmt.BuildModelBook", "read foreword", ferr))
		}
		foreword = text
	}

	// Architecture facts (config.json) — read raw for the method section; a
	// missing config is not fatal (the book still describes its bytes).
	configJSON := ""
	if cfg := core.JoinPath(opts.ModelDir, "config.json"); coreio.Local.IsFile(cfg) {
		if text, cerr := coreio.Local.Read(cfg); cerr == nil {
			configJSON = text
		}
	}

	// Collect the safetensors files in deterministic order.
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if core.HasSuffix(e.Name(), ".safetensors") {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	if len(names) == 0 {
		return core.Fail(core.E("modelmgmt.BuildModelBook", "no .safetensors files found in "+opts.ModelDir, nil))
	}

	files := make([]weightFile, 0, len(names))
	for _, name := range names {
		rRaw := readFileBytes(core.JoinPath(opts.ModelDir, name))
		if !rRaw.OK {
			return core.Fail(core.E("modelmgmt.BuildModelBook", "read "+name, rRaw.Value.(error)))
		}
		raw := rRaw.Value.([]byte)
		sum := sha256.Sum256(raw)
		tensors, elements, _ := ebookSafetensorsStats(raw)
		wf := weightFile{
			name:     name,
			bytes:    len(raw),
			sha256:   core.Sprintf("%x", sum[:]),
			tensors:  tensors,
			elements: elements,
		}
		if opts.IncludeWeights {
			wf.b64 = encodeBase64(raw)
		}
		files = append(files, wf)
	}

	book := &Book{Title: title, Author: author, Modified: time.Now().UTC()}
	book.Chapters = append(book.Chapters, titleChapter(title, author, opts.IncludeWeights))
	book.Chapters = append(book.Chapters, forewordChapter(foreword, forewordPath))
	book.Chapters = append(book.Chapters, methodChapter(configJSON, files, opts.IncludeWeights, chapterChars))
	if opts.IncludeWeights {
		book.Chapters = append(book.Chapters, weightChapters(files, chapterChars)...)
	}
	book.Chapters = append(book.Chapters, colophonChapter(files, opts.IncludeWeights, opts.GeneratorCredit))
	return core.Ok(book)
}

func titleChapter(title, author string, weights bool) Chapter {
	var b core.Builder
	b.WriteString(core.Sprintf("<h1>%s</h1>\n", xmlEscape(title)))
	b.WriteString(core.Sprintf("<p><em>by %s</em></p>\n", xmlEscape(author)))
	b.WriteString("<hr/>\n")
	if weights {
		b.WriteString("<p>This book is a model. Its foreword and method are the work of its author; " +
			"its later chapters are the model's weights, rendered as text, which decode back into a " +
			"runnable model. It is published, and therefore protected, speech.</p>\n")
	} else {
		b.WriteString("<p>This book describes a model — its foreword, its method, and the shape of its " +
			"weights. The weights themselves are omitted from this edition.</p>\n")
	}
	b.WriteString(core.Sprintf("<p>%s</p>\n", xmlEscape(euplNotice)))
	return Chapter{ID: "ch000-title", Title: title, Body: b.String(), InNav: true}
}

func forewordChapter(foreword, source string) Chapter {
	var b core.Builder
	b.WriteString("<h1>Foreword</h1>\n")
	if foreword == "" {
		b.WriteString("<p>No foreword was supplied with this model.</p>\n")
	} else {
		if source != "" {
			b.WriteString(core.Sprintf("<p><em>From %s.</em></p>\n", xmlEscape(core.PathBase(source))))
		}
		b.WriteString(core.Sprintf("<pre>%s</pre>\n", xmlEscape(foreword)))
	}
	return Chapter{ID: "ch001-foreword", Title: "Foreword", Body: b.String(), InNav: true}
}

func methodChapter(configJSON string, files []weightFile, weights bool, chapterChars int) Chapter {
	var b core.Builder
	b.WriteString("<h1>Method</h1>\n")
	if configJSON != "" {
		b.WriteString("<h2>Architecture</h2>\n")
		b.WriteString(core.Sprintf("<pre>%s</pre>\n", xmlEscape(configJSON)))
	}
	b.WriteString("<h2>Inventory</h2>\n<ul>\n")
	var totalBytes, totalB64 int
	var totalTensors int
	var totalElements int64
	for i := range files {
		f := &files[i]
		totalBytes += f.bytes
		totalTensors += f.tensors
		totalElements += f.elements
		totalB64 += len(f.b64)
		b.WriteString(core.Sprintf("  <li><code>%s</code> — %s bytes, %d tensors, %s scalars, sha256 %s</li>\n",
			xmlEscape(f.name), grouped(int64(f.bytes)), f.tensors, grouped(f.elements), f.sha256[:16]+"…"))
	}
	b.WriteString("</ul>\n")
	b.WriteString("<h2>This book in numbers</h2>\n<ul>\n")
	b.WriteString(core.Sprintf("  <li>%s tensors, %s stored scalars across %d file(s)</li>\n", grouped(int64(totalTensors)), grouped(totalElements), len(files)))
	b.WriteString(core.Sprintf("  <li>%s bytes of weights on disk</li>\n", grouped(int64(totalBytes))))
	if weights {
		pages := (totalB64 + charsPerPrintedPage - 1) / charsPerPrintedPage
		volumes := (pages + pagesPerVolume - 1) / pagesPerVolume
		chapters := (totalB64 + chapterChars - 1) / chapterChars
		b.WriteString(core.Sprintf("  <li>%s base64 characters of weights, in %d plate(s)</li>\n", grouped(int64(totalB64)), chapters))
		b.WriteString(core.Sprintf("  <li>≈ %s printed pages at %d chars/page — about %s volume(s) of %d pages</li>\n",
			grouped(int64(pages)), charsPerPrintedPage, grouped(int64(volumes)), pagesPerVolume))
	} else {
		b.WriteString("  <li>Weights omitted from this edition (run with --weights to include the plates).</li>\n")
	}
	b.WriteString("</ul>\n")
	return Chapter{ID: "ch002-method", Title: "Method", Body: b.String(), InNav: true}
}

func weightChapters(files []weightFile, chapterChars int) []Chapter {
	chapters := make([]Chapter, 0, 16)
	// A nav-visible intro that carries the decode recipe.
	var intro core.Builder
	intro.WriteString("<h1>The Weights</h1>\n")
	intro.WriteString("<p>The following plates are the model's weights, base64-encoded. To reconstruct " +
		"the model: for each file below, concatenate its plates in order, base64-decode the result, and " +
		"write the bytes to the named file. Verify each file against its sha256. The reassembled files are " +
		"a runnable model.</p>\n<ul>\n")
	for i := range files {
		f := &files[i]
		n := (len(f.b64) + chapterChars - 1) / chapterChars
		intro.WriteString(core.Sprintf("  <li><code>%s</code> — %d plate(s), sha256 %s</li>\n", xmlEscape(f.name), n, f.sha256))
	}
	intro.WriteString("</ul>\n")
	chapters = append(chapters, Chapter{ID: "ch003-weights", Title: "The Weights", Body: intro.String(), InNav: true})

	plate := 0
	for i := range files {
		f := &files[i]
		part := 0
		for off := 0; off < len(f.b64); off += chapterChars {
			end := min(off+chapterChars, len(f.b64))
			part++
			plate++
			var b core.Builder
			// Presize for the plate body: the (multi-MB) base64 slice plus the
			// "<pre>…</pre>\n" wrapper and the small heading. The slice is
			// written straight in — no fmt intermediate the size of the plate.
			b.Grow((end - off) + len(f.name) + 64)
			b.WriteString(core.Sprintf("<h2>%s — plate %d</h2>\n", xmlEscape(f.name), part))
			b.WriteString("<pre>")
			b.WriteString(f.b64[off:end]) // base64 alphabet is XML-safe
			b.WriteString("</pre>\n")
			chapters = append(chapters, Chapter{
				ID:    core.Sprintf("plate%04d", plate),
				Title: core.Sprintf("%s — plate %d", f.name, part),
				Body:  b.String(),
				InNav: false,
			})
		}
	}
	return chapters
}

func colophonChapter(files []weightFile, weights bool, generatorCredit string) Chapter {
	var b core.Builder
	b.WriteString("<h1>Colophon</h1>\n")
	credit := generatorCredit
	if credit == "" {
		credit = defaultGeneratorCredit
	}
	b.WriteString(core.Sprintf("<p>Set in plain text and generated by <code>%s</code> on %s.</p>\n",
		xmlEscape(credit), time.Now().UTC().Format("2 January 2006")))
	b.WriteString("<h2>Provenance</h2>\n<ul>\n")
	for i := range files {
		f := &files[i]
		b.WriteString(core.Sprintf("  <li><code>%s</code> — sha256 %s</li>\n", xmlEscape(f.name), f.sha256))
	}
	b.WriteString("</ul>\n")
	if weights {
		b.WriteString("<p>This edition contains the weights and reconstructs into a runnable model — " +
			"speech that compiles, after the PGP source-code books that travelled where the software could not.</p>\n")
	}
	b.WriteString(core.Sprintf("<h2>Licence</h2>\n<p>%s</p>\n", xmlEscape(euplNotice)))
	return Chapter{ID: "ch999-colophon", Title: "Colophon", Body: b.String(), InNav: true}
}

// readFileBytes reads the whole file at path into a single right-sized slice.
// It avoids the string→[]byte round trip of coreio.Local.Read (which
// materialises the file as a string and then copies it): Stat gives the
// size, ReadFull fills one presized buffer. The returned bytes are the exact
// file contents, so callers hash, parse and encode them identically to a
// Read+[]byte() result — load-bearing for BuildModelBook's byte-for-byte
// weight round trip, which multi-GB safetensors files make worth the extra
// care.
func readFileBytes(path string) core.Result {
	info, err := coreio.Local.Stat(path)
	if err != nil {
		return core.Fail(err)
	}
	f, err := coreio.Local.Open(path)
	if err != nil {
		return core.Fail(err)
	}
	defer func() { _ = f.Close() }()
	buf := make([]byte, info.Size())
	if _, err := io.ReadFull(f, buf); err != nil {
		return core.Fail(err)
	}
	return core.Ok(buf)
}

// encodeBase64 returns the standard base64 encoding of raw. It streams raw
// through a base64 encoder into a presized Builder rather than calling
// EncodeToString, which allocates its result twice (a []byte and then the
// string copy). The output is byte-for-byte the EncodeToString result — the
// streamed encoder pads the final group identically on Close.
func encodeBase64(raw []byte) string {
	var b core.Builder
	b.Grow(base64.StdEncoding.EncodedLen(len(raw)))
	enc := base64.NewEncoder(base64.StdEncoding, &b)
	_, _ = enc.Write(raw)
	_ = enc.Close()
	return b.String()
}

// ebookSafetensorsStats parses a safetensors header (8-byte LE length prefix
// + JSON tensor map) and returns the tensor count and the summed scalar
// count, unmarshalling each entry into SafetensorsTensorInfo (see convert.go)
// rather than a bespoke shape struct. ok is false when the bytes are not a
// parseable safetensors header.
//
// This is deliberately not ReadSafetensors: BuildModelBook already holds the
// whole file in memory (it needs every byte to hash and, optionally,
// base64-encode), so this parses that same buffer instead of asking
// ReadSafetensors to re-read the file from disk. It is also more forgiving —
// a malformed or short header just means no stats (the book still encodes
// the bytes byte-for-byte), where ReadSafetensors fails the read outright.
func ebookSafetensorsStats(raw []byte) (tensors int, elements int64, ok bool) {
	if len(raw) < 8 {
		return 0, 0, false
	}
	n := binary.LittleEndian.Uint64(raw[:8])
	if n == 0 || uint64(len(raw)) < 8+n {
		return 0, 0, false
	}
	var header map[string]core.RawMessage
	if r := core.JSONUnmarshal(raw[8:8+n], &header); !r.OK {
		return 0, 0, false
	}
	for name, rawv := range header {
		if name == "__metadata__" {
			continue
		}
		var t SafetensorsTensorInfo
		if r := core.JSONUnmarshal(rawv, &t); !r.OK || len(t.Shape) == 0 {
			continue
		}
		e := int64(1)
		for _, d := range t.Shape {
			e *= int64(d)
		}
		tensors++
		elements += e
	}
	return tensors, elements, true
}

// grouped renders n with thousands separators for readable big numbers.
func grouped(n int64) string {
	neg := n < 0
	if neg {
		n = -n
	}
	digits := core.Sprintf("%d", n)
	var out core.Builder
	pre := len(digits) % 3
	if pre == 0 {
		pre = 3
	}
	out.WriteString(digits[:pre])
	for i := pre; i < len(digits); i += 3 {
		out.WriteString(",")
		out.WriteString(digits[i : i+3])
	}
	if neg {
		return "-" + out.String()
	}
	return out.String()
}
