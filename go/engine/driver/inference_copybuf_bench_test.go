// SPDX-Licence-Identifier: EUPL-1.2

package driver

import (
	"io"
	"sync"
	"testing"
)

// These benchmarks isolate the per-request streaming-copy buffer in
// InferenceProvider.forward (inference.go:150), which today does a fresh
// `buf := make([]byte, 16*1024)` on every chat request. The pool variant
// proves the alloc the forward proxy pays per request can be eliminated.
//
// Modelled on a 64KB SSE response (≈ a short completion stream) copied in
// 16KB reads — the production loop shape.

const benchCopyChunk = 16 * 1024
const benchRespBytes = 64 * 1024

// forwardCopyMake mirrors the current forward() copy loop: allocate a 16KB
// buffer per call, copy the response through it.
func forwardCopyMake(dst io.Writer, src io.Reader) int {
	buf := make([]byte, benchCopyChunk)
	total := 0
	for {
		n, rerr := src.Read(buf)
		if n > 0 {
			_, _ = dst.Write(buf[:n])
			total += n
		}
		if rerr != nil {
			break
		}
	}
	return total
}

var forwardCopyPool = sync.Pool{New: func() any { b := make([]byte, benchCopyChunk); return &b }}

// forwardCopyPooled is the proposed shape: borrow the copy buffer from a pool.
func forwardCopyPooled(dst io.Writer, src io.Reader) int {
	bp := forwardCopyPool.Get().(*[]byte)
	buf := *bp
	defer forwardCopyPool.Put(bp)
	total := 0
	for {
		n, rerr := src.Read(buf)
		if n > 0 {
			_, _ = dst.Write(buf[:n])
			total += n
		}
		if rerr != nil {
			break
		}
	}
	return total
}

type benchZeroReader struct{ remaining int }

func (r *benchZeroReader) Read(p []byte) (int, error) {
	if r.remaining <= 0 {
		return 0, io.EOF
	}
	n := min(len(p), r.remaining)
	r.remaining -= n
	return n, nil
}

type benchDiscardWriter struct{}

func (benchDiscardWriter) Write(p []byte) (int, error) { return len(p), nil }

// BenchmarkForwardCopy_Make measures the current make-per-request shape — one
// 16KB heap allocation booked on every proxied chat request.
func BenchmarkForwardCopy_Make(b *testing.B) {
	w := benchDiscardWriter{}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = forwardCopyMake(w, &benchZeroReader{remaining: benchRespBytes})
	}
}

// BenchmarkForwardCopy_Pooled measures the sync.Pool variant — the copy buffer
// is reused, so the per-request 16KB alloc disappears.
func BenchmarkForwardCopy_Pooled(b *testing.B) {
	w := benchDiscardWriter{}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = forwardCopyPooled(w, &benchZeroReader{remaining: benchRespBytes})
	}
}
