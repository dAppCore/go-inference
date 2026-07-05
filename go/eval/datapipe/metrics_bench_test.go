// SPDX-Licence-Identifier: EUPL-1.2

package datapipe

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/store"
)

// seedStoreGolden seeds a store.DuckDB golden_set table with `rows` rows for the
// metrics/seed benchmarks.
func seedStoreGolden(b *testing.B, rows int) *store.DuckDB {
	b.Helper()
	db := newStoreDuckDBB(b)
	if _, err := db.Conn().Exec(`CREATE TABLE golden_set (
		idx INTEGER, seed_id VARCHAR, domain VARCHAR, voice VARCHAR,
		prompt VARCHAR, response VARCHAR, gen_time DOUBLE, char_count INTEGER
	)`); err != nil {
		b.Fatalf("create golden_set: %v", err)
	}
	if _, err := db.Conn().Exec(`INSERT INTO golden_set
		SELECT i, 'seed-' || i, 'domain-' || (i % 8), 'voice-' || (i % 5),
		       'prompt ' || i, 'response body ' || i, 1.25, 256
		FROM range(?) t(i)`, rows); err != nil {
		b.Fatalf("seed golden_set: %v", err)
	}
	return db
}

func newStoreDuckDBB(b *testing.B) *store.DuckDB {
	b.Helper()
	db, err := store.OpenDuckDBReadWrite(core.JoinPath(b.TempDir(), "store.duckdb"))
	requireResultOK(b, err)
	b.Cleanup(func() { _ = db.Close() })
	return db
}

func BenchmarkPushMetrics(b *testing.B) {
	db := seedStoreGolden(b, 500)
	influx, _ := newFakeInflux(b, nil, 0)
	sink := core.NewBuffer(nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink.Reset()
		if r := PushMetrics(db, influx, sink); !r.OK {
			b.Fatalf("push: %s", r.Error())
		}
	}
}
