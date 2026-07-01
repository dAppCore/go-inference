// SPDX-Licence-Identifier: EUPL-1.2

package modelmgmt

import (
	"encoding/base64"
	"encoding/binary"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// tinySafetensors builds a valid safetensors blob: one F32 tensor of shape
// [2,3] (6 scalars, 24 bytes of data).
func tinySafetensors() []byte {
	header := `{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}`
	prefix := make([]byte, 8)
	binary.LittleEndian.PutUint64(prefix, uint64(len(header)))
	out := append(prefix, []byte(header)...)
	data := make([]byte, 24)
	for i := range data {
		data[i] = byte(i * 7)
	}
	return append(out, data...)
}

// writeEbookFixtureModel writes a model directory shaped for BuildModelBook
// (config.json + README.md + one safetensors file) — distinct from
// writeSafetensorsFixture (gguf_test.go/convert_test.go), which builds a bare
// LoRA-adapter pair, not a full model directory.
func writeEbookFixtureModel(t *core.T) (dir string, weights []byte) {
	t.Helper()
	dir = core.JoinPath(t.TempDir(), "LEM-Tiny")
	core.RequireNoError(t, coreio.Local.EnsureDir(dir))
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(dir, "config.json"), `{"model_type":"gemma3_text","hidden_size":1152}`))
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(dir, "README.md"), "# LEM-Tiny\nThe loyal one.\n"))
	weights = tinySafetensors()
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(dir, "model.safetensors"), string(weights)))
	return dir, weights
}

// plateBase64 concatenates the base64 out of every plate chapter, in order —
// the reconstruction a reader would perform.
func plateBase64(chapters []Chapter) string {
	var out core.Builder
	for i := range chapters {
		ch := &chapters[i]
		if !core.HasPrefix(ch.ID, "plate") {
			continue
		}
		body := ch.Body
		start := core.Index(body, "<pre>")
		end := core.Index(body, "</pre>")
		if start < 0 || end < 0 {
			continue
		}
		out.WriteString(body[start+len("<pre>") : end])
	}
	return out.String()
}

// The load-bearing test: the weights survive the round trip (decode the
// plates back and you have the original safetensors, byte for byte — speech
// that compiles), and the no-weights edition omits the plates entirely.
func TestEbookModel_BuildModelBook_Good(t *core.T) {
	dir, weights := writeEbookFixtureModel(t)

	r := BuildModelBook(ModelBookOptions{ModelDir: dir, IncludeWeights: true, ChapterChars: 16})
	requireResultOK(t, r)
	book := r.Value.(*Book)
	core.AssertEqual(t, "LEM-Tiny", book.Title)

	decoded, err := base64.StdEncoding.DecodeString(plateBase64(book.Chapters))
	core.RequireNoError(t, err)
	core.AssertEqual(t, weights, decoded)

	// Byte-identity at the time-independent layer: the concatenated plate
	// base64 must equal the canonical EncodeToString of the original weights.
	// (The whole EPUB embeds wall-clock timestamps and is not reproducible; the
	// plate base64 is a pure function of the input bytes, and is what the
	// protected-speech reproducibility claim rests on.)
	core.AssertEqual(t, base64.StdEncoding.EncodeToString(weights), plateBase64(book.Chapters))

	// Small ChapterChars must split into several plates (proves chunking).
	plates := 0
	for i := range book.Chapters {
		if core.HasPrefix(book.Chapters[i].ID, "plate") {
			plates++
		}
	}
	if plates < 2 {
		t.Fatalf("plates = %d, want >1 with ChapterChars=16", plates)
	}

	// Foreword carried the README.
	core.AssertContains(t, book.Chapters[1].Body, "The loyal one.")

	// The no-weights edition is the manifesto + method only.
	rNo := BuildModelBook(ModelBookOptions{ModelDir: dir, IncludeWeights: false})
	requireResultOK(t, rNo)
	noBook := rNo.Value.(*Book)
	for i := range noBook.Chapters {
		if core.HasPrefix(noBook.Chapters[i].ID, "plate") {
			t.Fatal("no-weights edition must contain no plates")
		}
	}
	core.AssertContains(t, noBook.Chapters[2].Body, "omitted")
}

func TestEbookModel_BuildModelBook_Bad(t *core.T) {
	dir := core.JoinPath(t.TempDir(), "empty")
	core.RequireNoError(t, coreio.Local.EnsureDir(dir))
	assertResultError(t, BuildModelBook(ModelBookOptions{ModelDir: dir, IncludeWeights: true}), "no .safetensors")
}

// A malformed/sub-8-byte .safetensors is not a parseable header, but it is
// still bytes — the book must encode it (no stats, no panic) and round-trip
// it exactly, just like a valid file. This guards the streamed read/encode
// path against the short-file edge case.
func TestEbookModel_BuildModelBook_Ugly(t *core.T) {
	dir := core.JoinPath(t.TempDir(), "LEM-Malformed")
	core.RequireNoError(t, coreio.Local.EnsureDir(dir))
	garbage := []byte{1, 2, 3} // < 8 bytes: not a safetensors header
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(dir, "model.safetensors"), string(garbage)))
	r := BuildModelBook(ModelBookOptions{ModelDir: dir, IncludeWeights: true, ChapterChars: 16})
	requireResultOK(t, r)
	book := r.Value.(*Book)
	core.AssertEqual(t, base64.StdEncoding.EncodeToString(garbage), plateBase64(book.Chapters))
}

func TestEbookModel_ebookSafetensorsStats_Good(t *core.T) {
	tensors, elements, ok := ebookSafetensorsStats(tinySafetensors())
	core.AssertTrue(t, ok)
	core.AssertEqual(t, 1, tensors)
	core.AssertEqual(t, int64(6), elements)
}

// Garbage must not parse as a safetensors header — it is tolerated (ok=false)
// rather than treated as an error.
func TestEbookModel_ebookSafetensorsStats_Ugly(t *core.T) {
	_, _, ok := ebookSafetensorsStats([]byte{1, 2, 3})
	core.AssertFalse(t, ok)
}

func TestEbookModel_grouped_Good(t *core.T) {
	cases := map[int64]string{0: "0", 42: "42", 1000: "1,000", 999888777: "999,888,777"}
	for n, want := range cases {
		core.AssertEqual(t, want, grouped(n))
	}
}

func TestEbookModel_grouped_Ugly(t *core.T) {
	core.AssertEqual(t, "-1,234", grouped(-1234))
}
