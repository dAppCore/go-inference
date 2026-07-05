// SPDX-Licence-Identifier: EUPL-1.2

package modelmgmt

import (
	"archive/zip"
	"bytes"
	"io"

	core "dappco.re/go"
)

// readZipEntry reads one named entry out of an EPUB zip, failing the test if
// it is absent.
func readZipEntry(t *core.T, zr *zip.Reader, name string) string {
	t.Helper()
	for _, f := range zr.File {
		if f.Name == name {
			rc, err := f.Open()
			core.RequireNoError(t, err)
			defer rc.Close()
			data, err := io.ReadAll(rc)
			core.RequireNoError(t, err)
			return string(data)
		}
	}
	t.Fatalf("entry %s not found in epub", name)
	return ""
}

// A valid EPUB3: mimetype first and stored, container points at the OPF, the
// OPF carries the metadata + every chapter, nav lists only nav chapters, and
// each chapter is its own well-formed file.
func TestEbook_WriteEPUB_Good(t *core.T) {
	b := &Book{
		Title:  "LEM & friends <test>",
		Author: "Lethean",
		Chapters: []Chapter{
			{ID: "ch000", Title: "Title", Body: "<h1>Hi</h1>", InNav: true},
			{ID: "plate0001", Title: "Plate 1", Body: "<pre>QUJD</pre>", InNav: false},
			{ID: "ch999", Title: "Colophon", Body: "<p>end</p>", InNav: true},
		},
	}
	var buf bytes.Buffer
	requireResultOK(t, b.WriteEPUB(&buf))

	zr, err := zip.NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	core.RequireNoError(t, err)

	// mimetype MUST be the first entry, stored, exact content.
	if zr.File[0].Name != "mimetype" {
		t.Fatalf("first entry = %q, want mimetype", zr.File[0].Name)
	}
	if zr.File[0].Method != zip.Store {
		t.Fatal("mimetype must be stored uncompressed")
	}
	core.AssertEqual(t, epubMimetype, readZipEntry(t, zr, "mimetype"))

	readZipEntry(t, zr, "META-INF/container.xml")
	opf := readZipEntry(t, zr, "OEBPS/content.opf")
	for _, want := range []string{"<dc:title>LEM &amp; friends &lt;test&gt;</dc:title>", "<dc:rights>EUPL-1.2</dc:rights>", `idref="plate0001"`, "dcterms:modified"} {
		core.AssertContains(t, opf, want)
	}
	nav := readZipEntry(t, zr, "OEBPS/nav.xhtml")
	core.AssertNotContains(t, nav, "plate0001")
	core.AssertContains(t, nav, "ch999.xhtml")
	readZipEntry(t, zr, "OEBPS/ch000.xhtml")
	readZipEntry(t, zr, "OEBPS/plate0001.xhtml")
}

func TestEbook_WriteEPUB_Bad(t *core.T) {
	var buf bytes.Buffer
	assertResultError(t, (&Book{Title: "x"}).WriteEPUB(&buf), "at least one chapter")
}

// A book where every chapter is InNav:false still renders a valid (if
// table-of-contents-empty) container — WriteEPUB has no opinion on whether a
// book chooses to expose any navigation entries.
func TestEbook_WriteEPUB_Ugly(t *core.T) {
	b := &Book{
		Title:    "No ToC",
		Chapters: []Chapter{{ID: "plate0001", Title: "Plate 1", Body: "<pre>x</pre>", InNav: false}},
	}
	var buf bytes.Buffer
	requireResultOK(t, b.WriteEPUB(&buf))
	zr, err := zip.NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	core.RequireNoError(t, err)
	nav := readZipEntry(t, zr, "OEBPS/nav.xhtml")
	core.AssertNotContains(t, nav, "plate0001")
}

func TestEbook_xmlEscape_Good(t *core.T) {
	core.AssertEqual(t, "a &amp; b &lt; c &gt; d", xmlEscape("a & b < c > d"))
}

// Ampersand first — no double-escaping of the entities it introduces.
func TestEbook_xmlEscape_Ugly(t *core.T) {
	core.AssertEqual(t, "&lt;&amp;&gt;", xmlEscape("<&>"))
}
