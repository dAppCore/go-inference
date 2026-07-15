// SPDX-Licence-Identifier: EUPL-1.2

package phrasescan

import "testing"

// phrasescan_bench_test.go pins the scan cost of Count — the primitive the eval
// scorers run once per generated response over the whole text for every marker
// set (compliance + emotion). The two benches separate the clean hot path (no
// phrase begins here, single canStart reject per byte) from the matching path
// (a bucket probe plus \b assertions per hit): Count allocates nothing on either
// path, so allocs/op is expected to be 0 and the benches guard that.

// benchClean has no marker phrase, so every ASCII byte takes the canStart
// fast-reject — the common shape of ordinary model prose.
const benchClean = "the quick brown fox jumps over the lazy dog while the sun sets " +
	"slowly behind the distant hills and the river runs on toward the sea "

// benchMatching is seeded with phrases from testPhrases at several positions so
// the bucket-probe + boundary path is exercised, including a fold hazard.
const benchMatching = "as an ai i cannot feel deep sorrow but i respond responsibly " +
	"with kindneſs about this language model and i can't pretend otherwise "

// BenchmarkCount_Clean measures the fast-reject path on text with no matches.
func BenchmarkCount_Clean(b *testing.B) {
	ps := New(testPhrases)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ps.Count(benchClean)
	}
}

// BenchmarkCount_Matching measures the scan when several phrases fire — the
// bucket probe, literal match, and \b assertions on each hit.
func BenchmarkCount_Matching(b *testing.B) {
	ps := New(testPhrases)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ps.Count(benchMatching)
	}
}
