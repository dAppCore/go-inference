// SPDX-Licence-Identifier: EUPL-1.2

package datapipe

import (
	"testing"

	core "dappco.re/go"
)

func BenchmarkSeedInflux(b *testing.B) {
	db := seedStoreGolden(b, 500)
	// Force re-seed each iteration; the gold_gen DISTINCT-count query returns
	// nothing from the fake influx, so the write path always runs.
	influx, _ := newFakeInflux(b, nil, 0)
	cfg := SeedInfluxConfig{Force: true, BatchSize: 1 << 30}
	sink := core.NewBuffer(nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink.Reset()
		if r := SeedInflux(db, influx, cfg, sink); !r.OK {
			b.Fatalf("seed: %s", r.Error())
		}
	}
}
