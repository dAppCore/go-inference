// SPDX-Licence-Identifier: EUPL-1.2

package datapipe

import (
	"testing"

	core "dappco.re/go"
)

var (
	benchResult core.Result
)

// seedBenchDB builds a DuckDB with `rows` golden_set and expansion_prompts rows
// so the scan loops iterate over a realistic record count.
func seedBenchDB(b *testing.B, rows int) *DB {
	b.Helper()
	rDB := OpenDBReadWrite(core.JoinPath(b.TempDir(), "bench.duckdb"))
	requireResultOK(b, rDB)
	db := rDB.Value.(*DB)
	b.Cleanup(func() { _ = db.Close() })

	requireResultOK(b, db.Exec(`CREATE TABLE golden_set (
		idx INTEGER, seed_id VARCHAR, domain VARCHAR, voice VARCHAR,
		prompt VARCHAR, response VARCHAR, gen_time DOUBLE, char_count INTEGER
	)`))
	requireResultOK(b, db.Exec(`INSERT INTO golden_set
		SELECT i, 'seed-' || i, 'domain-' || (i % 8), 'voice-' || (i % 5),
		       'prompt text ' || i, 'a much longer response body for row ' || i,
		       1.25, 256
		FROM range(?) t(i)`, rows))

	requireResultOK(b, db.Exec(`CREATE TABLE expansion_prompts (
		idx BIGINT, seed_id VARCHAR, region VARCHAR, domain VARCHAR, language VARCHAR,
		prompt VARCHAR, prompt_en VARCHAR, priority INTEGER, status VARCHAR
	)`))
	requireResultOK(b, db.Exec(`INSERT INTO expansion_prompts
		SELECT i, 'seed-' || i, 'en', 'domain-' || (i % 8), 'en',
		       'prompt ' || i, 'prompt en ' || i, i % 3, 'pending'
		FROM range(?) t(i)`, rows))
	return db
}

func BenchmarkQueryGoldenSet(b *testing.B) {
	db := seedBenchDB(b, 1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResult = db.QueryGoldenSet(0)
		if !benchResult.OK {
			b.Fatalf("query: %s", benchResult.Error())
		}
	}
}

func BenchmarkQueryExpansionPrompts(b *testing.B) {
	db := seedBenchDB(b, 1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResult = db.QueryExpansionPrompts("pending", 0)
		if !benchResult.OK {
			b.Fatalf("query: %s", benchResult.Error())
		}
	}
}

func BenchmarkQueryRows(b *testing.B) {
	db := seedBenchDB(b, 1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResult = db.QueryRows("SELECT idx, seed_id, domain, voice, prompt, response, gen_time, char_count FROM golden_set ORDER BY idx")
		if !benchResult.OK {
			b.Fatalf("query: %s", benchResult.Error())
		}
	}
}

func BenchmarkCountGoldenSet(b *testing.B) {
	db := seedBenchDB(b, 1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResult = db.CountGoldenSet()
		if !benchResult.OK {
			b.Fatalf("count: %s", benchResult.Error())
		}
	}
}

func BenchmarkTableCounts(b *testing.B) {
	db := seedBenchDB(b, 100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResult = db.TableCounts()
		if !benchResult.OK {
			b.Fatalf("counts: %s", benchResult.Error())
		}
	}
}
