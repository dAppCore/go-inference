// SPDX-Licence-Identifier: EUPL-1.2

package modelmgmt

import (
	"archive/zip"
	"crypto/sha256"
	"io"
	"time"

	core "dappco.re/go"
)

// Chapter is one section of an authored EPUB3 work rendered by WriteEPUB.
// Every chapter is in the reading spine; InNav controls whether it also
// appears in the table of contents — BuildModelBook uses this so a book with
// hundreds of weight "plates" keeps a clean ToC while still being fully
// readable front to back.
type Chapter struct {
	ID    string // manifest id + file stem, e.g. "ch001"
	Title string
	Body  string // pre-escaped XHTML body inner markup
	InNav bool
}

// Book is an authored work ready to render as a valid EPUB3 container. It is
// engine-agnostic: nothing here depends on a loaded model or a GPU backend,
// only on the Chapters the caller supplies. BuildModelBook is one such
// caller, turning a model directory into a Book, but WriteEPUB itself has no
// opinion about where its content comes from — pure file transformation,
// fully testable without a model.
//
// Its reason to exist is the PGP playbook: Zimmermann printed PGP's source as
// a book because software was an exportable "munition" but a book is
// protected speech (Bernstein v. United States, Junger v. Daley settled that
// code is speech). A model rendered as an authored, published book carries
// the protection every published work carries — only a court, against the
// presumption and the burden, can strip it. EUPL-1.2 on the cover.
type Book struct {
	Title      string
	Author     string
	Language   string    // BCP-47; "" → "en"
	Identifier string    // unique dc:identifier; "" → derived from title+author
	Rights     string    // licence; "" → "EUPL-1.2"
	Modified   time.Time // dcterms:modified (required by EPUB3); zero → now
	Chapters   []Chapter
}

const epubMimetype = "application/epub+zip"

// WriteEPUB streams a valid EPUB3 container to w. The mimetype entry is
// written first and stored uncompressed, as the spec requires.
//
//	book := &modelmgmt.Book{Title: "…", Author: "…", Chapters: chapters}
//	f, _ := coreio.Local.Create("book.epub")
//	r := book.WriteEPUB(f)
//	if !r.OK { return r }
func (b *Book) WriteEPUB(w io.Writer) core.Result {
	if len(b.Chapters) == 0 {
		return core.Fail(core.E("modelmgmt.WriteEPUB", "a book needs at least one chapter", nil))
	}
	lang := b.Language
	if lang == "" {
		lang = "en"
	}
	rights := b.Rights
	if rights == "" {
		rights = "EUPL-1.2"
	}
	id := b.Identifier
	if id == "" {
		sum := sha256.Sum256([]byte(b.Title + "\x00" + b.Author))
		id = core.Sprintf("urn:lethean:ebook:%x", sum[:8])
	}
	modified := b.Modified
	if modified.IsZero() {
		modified = time.Now().UTC()
	}

	zw := zip.NewWriter(w)

	// mimetype — MUST be the first entry and stored uncompressed.
	mw, err := zw.CreateHeader(&zip.FileHeader{Name: "mimetype", Method: zip.Store})
	if err != nil {
		return core.Fail(core.E("modelmgmt.WriteEPUB", "create mimetype", err))
	}
	if _, err := io.WriteString(mw, epubMimetype); err != nil {
		return core.Fail(core.E("modelmgmt.WriteEPUB", "write mimetype", err))
	}

	if r := epubWrite(zw, "META-INF/container.xml", epubContainerXML); !r.OK {
		return r
	}
	if r := epubWrite(zw, "OEBPS/content.opf", b.opf(id, lang, rights, modified)); !r.OK {
		return r
	}
	if r := epubWrite(zw, "OEBPS/nav.xhtml", b.navXHTML()); !r.OK {
		return r
	}
	for i := range b.Chapters {
		if r := writeChapter(zw, &b.Chapters[i]); !r.OK {
			return r
		}
	}
	if err := zw.Close(); err != nil {
		return core.Fail(core.E("modelmgmt.WriteEPUB", "finalise epub", err))
	}
	return core.Ok(nil)
}

func epubWrite(zw *zip.Writer, name, content string) core.Result {
	fw, err := zw.Create(name)
	if err != nil {
		return core.Fail(core.E("modelmgmt.WriteEPUB", "create "+name, err))
	}
	if _, err := io.WriteString(fw, content); err != nil {
		return core.Fail(core.E("modelmgmt.WriteEPUB", "write "+name, err))
	}
	return core.Ok(nil)
}

const epubContainerXML = `<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
`

func (b *Book) opf(id, lang, rights string, modified time.Time) string {
	var out core.Builder
	out.WriteString(`<?xml version="1.0" encoding="UTF-8"?>` + "\n")
	out.WriteString(`<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">` + "\n")
	out.WriteString("  <metadata xmlns:dc=\"http://purl.org/dc/elements/1.1/\">\n")
	out.WriteString(core.Sprintf("    <dc:identifier id=\"bookid\">%s</dc:identifier>\n", xmlEscape(id)))
	out.WriteString(core.Sprintf("    <dc:title>%s</dc:title>\n", xmlEscape(b.Title)))
	out.WriteString(core.Sprintf("    <dc:creator>%s</dc:creator>\n", xmlEscape(b.Author)))
	out.WriteString(core.Sprintf("    <dc:language>%s</dc:language>\n", xmlEscape(lang)))
	out.WriteString(core.Sprintf("    <dc:rights>%s</dc:rights>\n", xmlEscape(rights)))
	out.WriteString(core.Sprintf("    <meta property=\"dcterms:modified\">%s</meta>\n", modified.Format("2006-01-02T15:04:05Z")))
	out.WriteString("  </metadata>\n  <manifest>\n")
	out.WriteString("    <item id=\"nav\" href=\"nav.xhtml\" media-type=\"application/xhtml+xml\" properties=\"nav\"/>\n")
	for i := range b.Chapters {
		ch := &b.Chapters[i]
		out.WriteString(core.Sprintf("    <item id=\"%s\" href=\"%s.xhtml\" media-type=\"application/xhtml+xml\"/>\n", ch.ID, ch.ID))
	}
	out.WriteString("  </manifest>\n  <spine>\n")
	for i := range b.Chapters {
		out.WriteString(core.Sprintf("    <itemref idref=\"%s\"/>\n", b.Chapters[i].ID))
	}
	out.WriteString("  </spine>\n</package>\n")
	return out.String()
}

func (b *Book) navXHTML() string {
	var out core.Builder
	out.WriteString(`<?xml version="1.0" encoding="UTF-8"?>` + "\n")
	out.WriteString(`<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">` + "\n")
	out.WriteString("<head><title>Contents</title></head>\n<body>\n  <nav epub:type=\"toc\" id=\"toc\">\n    <h1>Contents</h1>\n    <ol>\n")
	for i := range b.Chapters {
		ch := &b.Chapters[i]
		if !ch.InNav {
			continue
		}
		out.WriteString(core.Sprintf("      <li><a href=\"%s.xhtml\">%s</a></li>\n", ch.ID, xmlEscape(ch.Title)))
	}
	out.WriteString("    </ol>\n  </nav>\n</body>\n</html>\n")
	return out.String()
}

// writeChapter writes one chapter as its own XHTML file directly into the
// zip, streaming the (potentially multi-MB) body straight to the entry writer
// rather than first concatenating the whole document into an intermediate
// Builder — the plate bodies are large, and that intermediate was a second
// copy of every plate.
func writeChapter(zw *zip.Writer, ch *Chapter) core.Result {
	fw, err := zw.Create("OEBPS/" + ch.ID + ".xhtml")
	if err != nil {
		return core.Fail(core.E("modelmgmt.WriteEPUB", "create "+ch.ID+".xhtml", err))
	}
	for _, part := range []string{
		`<?xml version="1.0" encoding="UTF-8"?>` + "\n",
		`<html xmlns="http://www.w3.org/1999/xhtml">` + "\n",
		"<head><title>" + xmlEscape(ch.Title) + "</title></head>\n<body>\n",
		ch.Body,
		"\n</body>\n</html>\n",
	} {
		if _, err := io.WriteString(fw, part); err != nil {
			return core.Fail(core.E("modelmgmt.WriteEPUB", "write "+ch.ID+".xhtml", err))
		}
	}
	return core.Ok(nil)
}

// xmlEscape escapes the three load-bearing XML metacharacters for text
// content and attributes. Ampersand first, always, so the entities it
// introduces are not themselves re-escaped.
func xmlEscape(s string) string {
	s = core.Replace(s, "&", "&amp;")
	s = core.Replace(s, "<", "&lt;")
	s = core.Replace(s, ">", "&gt;")
	return s
}
