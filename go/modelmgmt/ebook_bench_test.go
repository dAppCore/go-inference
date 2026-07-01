// SPDX-Licence-Identifier: EUPL-1.2

package modelmgmt

import (
	"encoding/binary"
	"io"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// benchSafetensors builds a valid safetensors blob with a multi-MB data
// section so the base64/stream path dominates allocations the way a real
// weight file does (the 24-byte fixture in ebook_model_test.go is too small
// to move B/op).
func benchSafetensors(dataBytes int) []byte {
	header := `{"w":{"dtype":"F32","shape":[1,` + core.Sprintf("%d", dataBytes/4) + `],"data_offsets":[0,` + core.Sprintf("%d", dataBytes) + `]}}`
	prefix := make([]byte, 8)
	binary.LittleEndian.PutUint64(prefix, uint64(len(header)))
	out := make([]byte, 0, 8+len(header)+dataBytes)
	out = append(out, prefix...)
	out = append(out, header...)
	data := make([]byte, dataBytes)
	for i := range data {
		data[i] = byte(i*131 + 7)
	}
	out = append(out, data...)
	return out
}

// benchModelDir writes a synthetic model dir (README + config + one multi-MB
// safetensors file) and returns its path. Allocations here are excluded from
// the benchmark via b.ResetTimer in the callers.
func benchModelDir(b *testing.B, dataBytes int) string {
	b.Helper()
	dir := core.JoinPath(b.TempDir(), "LEM-Bench")
	if err := coreio.Local.EnsureDir(dir); err != nil {
		b.Fatalf("mkdir: %v", err)
	}
	if err := coreio.Local.Write(core.JoinPath(dir, "config.json"), `{"model_type":"gemma3_text","hidden_size":1152}`); err != nil {
		b.Fatalf("config: %v", err)
	}
	if err := coreio.Local.Write(core.JoinPath(dir, "README.md"), "# LEM-Bench\nThe loyal one.\n"); err != nil {
		b.Fatalf("readme: %v", err)
	}
	if err := coreio.Local.Write(core.JoinPath(dir, "model.safetensors"), string(benchSafetensors(dataBytes))); err != nil {
		b.Fatalf("weights: %v", err)
	}
	return dir
}

// BenchmarkBuildModelBook_Weights drives the heavy export path: read the
// safetensors file, hash it, render it to base64 plates. This is the
// B/op-dominant path (the weight bytes flow through it).
func BenchmarkBuildModelBook_Weights(b *testing.B) {
	dir := benchModelDir(b, 4<<20) // 4 MiB of weights
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := BuildModelBook(ModelBookOptions{ModelDir: dir, IncludeWeights: true, ChapterChars: defaultWeightChapterChars})
		if !r.OK {
			b.Fatalf("BuildModelBook: %s", r.Error())
		}
		if len(r.Value.(*Book).Chapters) == 0 {
			b.Fatal("no chapters")
		}
	}
}

// BenchmarkWriteEPUB_Weights drives the full render: build the weighted book
// once, then stream it to a discarding writer each iteration. Exercises the
// chapter-body assembly that copies the plate-sized strings into the zip.
func BenchmarkWriteEPUB_Weights(b *testing.B) {
	dir := benchModelDir(b, 4<<20)
	r := BuildModelBook(ModelBookOptions{ModelDir: dir, IncludeWeights: true, ChapterChars: defaultWeightChapterChars})
	if !r.OK {
		b.Fatalf("BuildModelBook: %s", r.Error())
	}
	book := r.Value.(*Book)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if r := book.WriteEPUB(io.Discard); !r.OK {
			b.Fatalf("WriteEPUB: %s", r.Error())
		}
	}
}
