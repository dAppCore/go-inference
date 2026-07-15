// SPDX-Licence-Identifier: EUPL-1.2

// Allocation contracts for the lab dashboard HTTP handlers (AX-11). index and
// healthz serve fixed bodies; requireAuth gates each request with a
// constant-time token compare. These endpoints are hit continuously by load
// balancers and liveness probes, so their per-request allocation matters. The
// benches use a no-op ResponseWriter and a pre-built request so each line
// isolates the handler's OWN per-request work from net/http's request plumbing.
//
// Run: go test -bench=. -benchmem -run='^$' ./lab/
package lab

import (
	"net/http"
	"testing"
)

// benchRW is an allocation-free ResponseWriter: Header returns one reused map,
// Write and WriteHeader discard. http.Header.Set still allocates its value
// slice (stdlib), which is the residual floor on index/healthz.
type benchRW struct{ hdr http.Header }

func (w *benchRW) Header() http.Header {
	if w.hdr == nil {
		w.hdr = make(http.Header, 1)
	}
	return w.hdr
}
func (w *benchRW) Write(b []byte) (int, error) { return len(b), nil }
func (w *benchRW) WriteHeader(int)             {}

var benchCalled int

func BenchmarkIndex(b *testing.B) {
	w := &benchRW{}
	r := &http.Request{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		index(w, r)
	}
}

func BenchmarkHealthz(b *testing.B) {
	w := &benchRW{}
	r := &http.Request{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		healthz(w, r)
	}
}

func BenchmarkRequireAuth(b *testing.B) {
	next := func(http.ResponseWriter, *http.Request) { benchCalled++ }
	h := requireAuth(next, "secret-bearer-token")
	w := &benchRW{}
	r := &http.Request{Header: http.Header{"Authorization": {"Bearer secret-bearer-token"}}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h(w, r)
	}
}
