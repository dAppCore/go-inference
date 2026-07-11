// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the admin cache-entries label parsing. Per AX-11 — this runs
// once per GET /v1/admin/cache/entries request (admin multiplier, not
// per-token): cacheEntryLabelsFrom filters an already-parsed query into the
// label map the cache lister filters on, skipping the reserved "model" key and
// empty values. The bench records the per-request map/label allocation profile.
//
// Run:    go test -bench=. -benchmem -run='^$' ./serving/compat/
package compat

import (
	"net/url"
	"testing"
)

// Sink defeats compiler DCE.
var compatBenchSinkLabels map[string]string

// benchLabelQuery is a realistic cache-entries filter: the reserved model key
// plus a handful of label filters, the shape an admin console sends.
func benchLabelQuery() url.Values {
	return url.Values{
		"model":    {"gemma-4-31b"},
		"tenant":   {"acme"},
		"workflow": {"support"},
		"lane":     {"draft"},
		"empty":    {""},
	}
}

func BenchmarkCacheEntryLabelsFrom(b *testing.B) {
	query := benchLabelQuery()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		compatBenchSinkLabels = cacheEntryLabelsFrom(query)
	}
}
