// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for agent package small utilities. These helpers fire on
// every wake/sleep round (firstNonEmpty inside loadIndex + SleepURIs,
// stateHash inside indexModel, cloneStringMap inside sleepEntryMeta).
//
// Per AX-11 — each individual call is sub-microsecond, but Sleep
// constructs a fresh map per invocation and stateHash hits a
// fmt.Sprintf chain; cumulative cost matters when the agent dispatches
// 100s of sleep rounds per session.
//
// Run:    go test -bench='BenchmarkHelpers' -benchmem -run='^$' ./go/agent

package agent

import (
	"testing"

	"dappco.re/go/inference/model/bundle"
)

// Sinks defeat compiler DCE.
var (
	helpersBenchSinkString string
	helpersBenchSinkMap    map[string]string
	helpersBenchSinkTok    bundle.Tokenizer
)

// --- firstNonEmpty — the trim+selectfirst loop. Fires inside
// loadIndex (one call per wake) and SleepURIs (3+ calls per sleep).

func BenchmarkHelpers_FirstNonEmpty_FirstHit(b *testing.B) {
	values := []string{"primary", "", "tertiary"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkString = firstNonEmpty(values...)
	}
}

func BenchmarkHelpers_FirstNonEmpty_LastHit(b *testing.B) {
	// Two empty/whitespace candidates before the real value — worst case
	// for the Trim loop.
	values := []string{"", "   ", "tertiary"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkString = firstNonEmpty(values...)
	}
}

func BenchmarkHelpers_FirstNonEmpty_AllEmpty(b *testing.B) {
	values := []string{"", "   ", ""}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkString = firstNonEmpty(values...)
	}
}

func BenchmarkHelpers_FirstNonEmptyString_LegacyAlias(b *testing.B) {
	values := []string{"", "fallback"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkString = firstNonEmptyString(values...)
	}
}

// --- stateHash — SHA-256 over a typical model identity string.
// Fired once per index build inside indexModel.

func BenchmarkHelpers_StateHash_ShortValue(b *testing.B) {
	value := "qwen3"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkString = stateHash(value)
	}
}

func BenchmarkHelpers_StateHash_ModelIdentity(b *testing.B) {
	// Composite identity string of the shape indexModel constructs —
	// name|path|arch|vocab|layers|quant|context.
	value := "qwen3-7b\n/models/qwen3-7b\nqwen3\n151936\n28\n4\n40960"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkString = stateHash(value)
	}
}

// --- stateBundleTokenizer — wrapper around bundle.NormaliseTokenizer.
// Hit once per index build.

func BenchmarkHelpers_StateBundleTokenizer_FullyPopulated(b *testing.B) {
	t := bundle.Tokenizer{
		Hash:             "deadbeef",
		ChatTemplateHash: "feed1234",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkTok = stateBundleTokenizer(t)
	}
}

func BenchmarkHelpers_StateBundleTokenizer_PathOnly(b *testing.B) {
	// Path set but no Hash — exercises the NormaliseTokenizer SHA path.
	t := bundle.Tokenizer{Path: "/tokenizers/qwen3-7b"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkTok = stateBundleTokenizer(t)
	}
}

// --- cloneStringMap — defensive copy of opts.Meta during sleep.
// Hit once per sleep round; cost is O(map size).

func BenchmarkHelpers_CloneStringMap_Nil(b *testing.B) {
	var src map[string]string
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkMap = cloneStringMap(src)
	}
}

func BenchmarkHelpers_CloneStringMap_Empty(b *testing.B) {
	src := map[string]string{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkMap = cloneStringMap(src)
	}
}

func BenchmarkHelpers_CloneStringMap_TypicalMeta(b *testing.B) {
	src := map[string]string{
		"agent":             "cladius",
		"session_id":        "s-3019c3b3",
		"parent_entry_uri":  "mlx://state/parent",
		"parent_bundle_uri": "mlx://state/parent/bundle",
		"parent_index_uri":  "mlx://state/parent/index",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkMap = cloneStringMap(src)
	}
}
