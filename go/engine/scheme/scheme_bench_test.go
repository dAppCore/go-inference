// SPDX-Licence-Identifier: EUPL-1.2

package scheme

import "testing"

// The scheme package is the engine's pluggable-component contract layer — three
// registries resolved at model load. The hot operations are registry lookups
// (interface→interface, 0 alloc), the mixer-owns-state Compatible check
// (branch-only), and the catalogue listers (Names() — a genuine output slice
// owned by core's registry). These benches pin those allocation profiles so a
// future change that regresses a lookup into a heap alloc is caught.

var (
	benchMixer Mixer
	benchCache CacheScheme
	benchQuant QuantScheme
	benchOK    bool
	benchBool  bool
	benchNames []string
)

func BenchmarkMixerFor(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchMixer, benchOK = MixerFor("softmax-hybrid")
	}
}

func BenchmarkCacheFor(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchCache, benchOK = CacheFor("q8")
	}
}

func BenchmarkQuantFor(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchQuant, benchOK = QuantFor("affine")
	}
}

func BenchmarkLookupMiss(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchMixer, benchOK = MixerFor("does-not-exist")
	}
}

func BenchmarkCompatible(b *testing.B) {
	m, _ := MixerFor("softmax-hybrid")
	c, _ := CacheFor("q8")
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchBool = Compatible(m, c)
	}
}

func BenchmarkStateKindString(b *testing.B) {
	var s string
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		s = StateKVCache.String()
	}
	_ = s
}

func BenchmarkMixerKinds(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchNames = MixerKinds()
	}
}

func BenchmarkCacheModes(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchNames = CacheModes()
	}
}
