// SPDX-Licence-Identifier: EUPL-1.2

package datapipe

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

var (
	benchFloat float64
	benchOK    bool
)

func BenchmarkExtractIteration(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchInt = extractIteration("gemma4-lek-philosophy@200")
	}
}

func BenchmarkToFloat64_Float(b *testing.B) {
	var v any = 8.5
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchFloat, benchOK = toFloat64(v)
	}
}

func BenchmarkToFloat64_String(b *testing.B) {
	var v any = "8.5"
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchFloat, benchOK = toFloat64(v)
	}
}

// buildContentScoresFixture writes a realistic content-scores JSONL file and
// returns its path. Each line carries several aggregate dimensions and a couple
// of probes, mirroring a benchmark scoring run.
func buildContentScoresFixture(b *testing.B, lines int) string {
	b.Helper()
	var sb core.Builder
	for i := 0; i < lines; i++ {
		sb.WriteString(`{"label":"gemma4-lek@`)
		sb.WriteString(core.Itoa(i))
		sb.WriteString(`","aggregates":{"clarity":8.5,"depth":7.25,"tone":9.0,"accuracy":6.75},`)
		sb.WriteString(`"probes":{"p001":{"scores":{"clarity":8.0,"depth":7.5}},"p002":{"scores":{"clarity":9.0,"depth":6.5}}}}` + "\n")
	}
	path := core.JoinPath(b.TempDir(), "content.jsonl")
	if err := coreio.Local.Write(path, sb.String()); err != nil {
		b.Fatalf("write fixture: %v", err)
	}
	return path
}

func buildCapabilityScoresFixture(b *testing.B, lines int) string {
	b.Helper()
	var sb core.Builder
	for i := 0; i < lines; i++ {
		sb.WriteString(`{"label":"gemma4-lek@`)
		sb.WriteString(core.Itoa(i))
		sb.WriteString(`","accuracy":0.85,"correct":85,"total":100,`)
		sb.WriteString(`"by_category":{"math":{"correct":40,"total":50},"logic":{"correct":45,"total":50},"ethics":{"correct":30,"total":40}}}` + "\n")
	}
	path := core.JoinPath(b.TempDir(), "capability.jsonl")
	if err := coreio.Local.Write(path, sb.String()); err != nil {
		b.Fatalf("write fixture: %v", err)
	}
	return path
}

func buildTrainingLogFixture(b *testing.B, lines int) string {
	b.Helper()
	var sb core.Builder
	for i := 0; i < lines; i++ {
		sb.WriteString("Iter ")
		sb.WriteString(core.Itoa(i))
		sb.WriteString(": Train loss 0.523, Learning Rate 1.0e-05, It/sec 2.15, Tokens/sec 30.42\n")
		if i%10 == 0 {
			sb.WriteString("Iter ")
			sb.WriteString(core.Itoa(i))
			sb.WriteString(": Val loss 0.601\n")
		}
	}
	path := core.JoinPath(b.TempDir(), "train.log")
	if err := coreio.Local.Write(path, sb.String()); err != nil {
		b.Fatalf("write fixture: %v", err)
	}
	return path
}

func BenchmarkIngestContentScores(b *testing.B) {
	path := buildContentScoresFixture(b, 200)
	influx, _ := newFakeInflux(b, nil, 0)
	cfg := IngestConfig{ContentFile: path, Model: "gemma4-lek", RunID: "run1", BatchSize: 1 << 30}
	sink := core.NewBuffer(nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink.Reset()
		if r := ingestContentScores(influx, cfg, sink); !r.OK {
			b.Fatalf("ingest: %s", r.Error())
		}
	}
}

func BenchmarkIngestCapabilityScores(b *testing.B) {
	path := buildCapabilityScoresFixture(b, 200)
	influx, _ := newFakeInflux(b, nil, 0)
	cfg := IngestConfig{CapabilityFile: path, Model: "gemma4-lek", RunID: "run1", BatchSize: 1 << 30}
	sink := core.NewBuffer(nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink.Reset()
		if r := ingestCapabilityScores(influx, cfg, sink); !r.OK {
			b.Fatalf("ingest: %s", r.Error())
		}
	}
}

func BenchmarkIngestTrainingLog(b *testing.B) {
	path := buildTrainingLogFixture(b, 400)
	influx, _ := newFakeInflux(b, nil, 0)
	cfg := IngestConfig{TrainingLog: path, Model: "gemma4-lek", RunID: "run1", BatchSize: 1 << 30}
	sink := core.NewBuffer(nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink.Reset()
		if r := ingestTrainingLog(influx, cfg, sink); !r.OK {
			b.Fatalf("ingest: %s", r.Error())
		}
	}
}
