// SPDX-Licence-Identifier: EUPL-1.2

package datapipe

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

func buildChatJSONLFixture(b *testing.B, lines int) string {
	b.Helper()
	var sb core.Builder
	for range lines {
		sb.WriteString(`{"messages":[`)
		sb.WriteString(`{"role":"system","content":"You are a calm philosophical assistant."},`)
		sb.WriteString(`{"role":"user","content":"Explain the moral imperative of consciousness."},`)
		sb.WriteString(`{"role":"assistant","content":"Consciousness protects consciousness; it enshrines that protection in its foundation."}`)
		sb.WriteString("]}\n")
	}
	path := core.JoinPath(b.TempDir(), "train.jsonl")
	if err := coreio.Local.Write(path, sb.String()); err != nil {
		b.Fatalf("write fixture: %v", err)
	}
	return path
}

func BenchmarkExportSplitParquet(b *testing.B) {
	path := buildChatJSONLFixture(b, 200)
	outDir := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if r := ExportSplitParquet(path, outDir, "train"); !r.OK {
			b.Fatalf("export: %s", r.Error())
		}
	}
}
